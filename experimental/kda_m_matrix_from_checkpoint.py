#!/usr/bin/env python3
from __future__ import annotations

"""Probe KDA M from a trained checkpoint (pure PyTorch reference path).

This script:
1) loads `final_model.pt`-style state_dict checkpoints,
2) runs a reference GPT forward up to a selected KDA layer,
3) extracts trained KDA tensors (k, g, beta, v),
4) builds transition matrix M and reports |M_ij| by diagonal distance.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:
    np = None


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _device_from_name(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            sd = obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            raise ValueError(f"Unsupported checkpoint structure at {path}")
    else:
        raise ValueError(f"Unsupported checkpoint object type: {type(obj)}")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    return sd


def _infer_arch(sd: dict[str, torch.Tensor]) -> dict[str, int | bool]:
    if "tok_emb.weight" not in sd:
        raise KeyError("Checkpoint is missing tok_emb.weight")

    block_re = re.compile(r"^blocks\.(\d+)\.")
    block_ids = sorted({int(m.group(1)) for k in sd for m in [block_re.match(k)] if m})
    if not block_ids:
        raise ValueError("No transformer block keys found (expected blocks.{i}.*)")

    num_layers = max(block_ids) + 1
    num_encoder_layers = num_layers // 2
    num_decoder_layers = num_layers - num_encoder_layers
    vocab_size, model_dim = sd["tok_emb.weight"].shape
    if "blocks.0.attn.kda.A_log" not in sd:
        raise ValueError("Checkpoint does not look like KDA attention (missing blocks.0.attn.kda.A_log)")

    num_heads = int(sd["blocks.0.attn.kda.A_log"].numel())
    key_dim = int(sd["blocks.0.attn.kda.k_proj.weight"].shape[0])
    value_dim = int(sd["blocks.0.attn.kda.v_proj.weight"].shape[0])
    if key_dim % num_heads != 0 or value_dim % num_heads != 0:
        raise ValueError("Inferred key/value dims are not divisible by num_heads")

    use_short_conv = "blocks.0.attn.kda.q_conv1d.weight" in sd
    return {
        "vocab_size": int(vocab_size),
        "model_dim": int(model_dim),
        "num_layers": int(num_layers),
        "num_encoder_layers": int(num_encoder_layers),
        "num_decoder_layers": int(num_decoder_layers),
        "num_heads": int(num_heads),
        "head_k_dim": int(key_dim // num_heads),
        "head_v_dim": int(value_dim // num_heads),
        "use_short_conv": bool(use_short_conv),
    }


def _cast_state(sd: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
    return out


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    bias = b.to(dtype=x.dtype) if b is not None else None
    return F.linear(x, w.to(dtype=x.dtype), bias)


def _rms_norm_last_dim(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


def _depthwise_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    # x: [B, T, D], weight: [D, 1, W]
    bsz, tlen, dim = x.shape
    wlen = weight.shape[-1]
    y = F.conv1d(
        x.transpose(1, 2),
        weight.to(dtype=x.dtype),
        bias.to(dtype=x.dtype) if bias is not None else None,
        padding=wlen - 1,
        groups=dim,
    )
    y = y[:, :, :tlen].transpose(1, 2).contiguous()
    if activation in {"silu", "swish"}:
        y = F.silu(y)
    return y


def _l2norm_last_dim(x: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv


def _naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = v.dtype
    bsz, tlen, heads, key_dim = q.shape
    value_dim = v.shape[-1]

    if scale is None:
        scale = key_dim ** -0.5
    if compute_dtype is None:
        compute_dtype = q.dtype if q.is_floating_point() else torch.float32

    q, k, v, g, beta = (x.to(compute_dtype) for x in (q, k, v, g, beta))
    q = q * scale
    s = k.new_zeros(bsz, heads, key_dim, value_dim)
    if initial_state is not None:
        s = s + initial_state.to(compute_dtype)

    o = torch.zeros_like(v)
    for i in range(tlen):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        s = s * g_i[..., None].exp()
        s = s + torch.einsum(
            "b h k, b h v -> b h k v",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * s).sum(-2),
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, s)

    if not output_final_state:
        s = None
    return o.to(dtype), s


def _kda_forward_reference(
    sd: dict[str, torch.Tensor],
    block_idx: int,
    hidden_states: torch.Tensor,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_short_conv: bool,
    lower_bound: float | None,
    allow_neg_eigval: bool,
    l2norm_eps: float,
    o_norm_eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p = f"blocks.{block_idx}.attn.kda."

    q_proj_w = sd[p + "q_proj.weight"]
    k_proj_w = sd[p + "k_proj.weight"]
    v_proj_w = sd[p + "v_proj.weight"]
    f0_w = sd[p + "f_proj.0.weight"]
    f1_w = sd[p + "f_proj.1.weight"]
    b_proj_w = sd[p + "b_proj.weight"]
    A_log = sd[p + "A_log"]
    dt_bias = sd[p + "dt_bias"]
    g0_w = sd[p + "g_proj.0.weight"]
    g1_w = sd[p + "g_proj.1.weight"]
    g1_b = sd.get(p + "g_proj.1.bias")
    o_norm_w = sd[p + "o_norm.weight"]
    o_proj_w = sd[p + "o_proj.weight"]

    if use_short_conv:
        q = _depthwise_causal_conv1d(
            _linear(hidden_states, q_proj_w),
            sd[p + "q_conv1d.weight"],
            sd.get(p + "q_conv1d.bias"),
            activation="silu",
        )
        k = _depthwise_causal_conv1d(
            _linear(hidden_states, k_proj_w),
            sd[p + "k_conv1d.weight"],
            sd.get(p + "k_conv1d.bias"),
            activation="silu",
        )
        v = _depthwise_causal_conv1d(
            _linear(hidden_states, v_proj_w),
            sd[p + "v_conv1d.weight"],
            sd.get(p + "v_conv1d.bias"),
            activation="silu",
        )
    else:
        q = F.silu(_linear(hidden_states, q_proj_w))
        k = F.silu(_linear(hidden_states, k_proj_w))
        v = F.silu(_linear(hidden_states, v_proj_w))

    g_raw = _linear(_linear(hidden_states, f0_w), f1_w)
    beta = torch.sigmoid(_linear(hidden_states, b_proj_w))

    bsz, tlen, _ = hidden_states.shape
    q = q.view(bsz, tlen, num_heads, head_k_dim)
    k = k.view(bsz, tlen, num_heads, head_k_dim)
    g_raw = g_raw.view(bsz, tlen, num_heads, head_k_dim)
    v = v.view(bsz, tlen, num_heads, head_v_dim)

    if allow_neg_eigval:
        beta = beta * 2.0

    # Match standard KDA training path (chunk/fused): internal q/k L2 normalization.
    q = _l2norm_last_dim(q, eps=l2norm_eps)
    k = _l2norm_last_dim(k, eps=l2norm_eps)

    dt = dt_bias.view(1, 1, num_heads, head_k_dim)
    a = torch.exp(A_log).view(1, 1, num_heads, 1)
    if lower_bound is None:
        g_log = -a * F.softplus(g_raw + dt)
    else:
        g_log = lower_bound * torch.sigmoid(a * (g_raw + dt))

    o, _ = _naive_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g_log,
        beta=beta,
        output_final_state=False,
        compute_dtype=hidden_states.dtype,
    )

    gate = _linear(_linear(hidden_states, g0_w), g1_w, g1_b).view(bsz, tlen, num_heads, head_v_dim)
    o = o / torch.sqrt((o * o).mean(dim=-1, keepdim=True) + o_norm_eps)
    o = o * o_norm_w.view(1, 1, 1, head_v_dim)
    o = o * torch.sigmoid(gate)
    o = o.reshape(bsz, tlen, num_heads * head_v_dim)
    o = _linear(o, o_proj_w)

    ctx = {
        "q": q.detach(),
        "k": k.detach(),
        "g": g_log.detach(),
        "beta": beta.detach(),
        "v": v.detach(),
    }
    return o, ctx


def _mlp_forward_reference(sd: dict[str, torch.Tensor], block_idx: int, x: torch.Tensor) -> torch.Tensor:
    p = f"blocks.{block_idx}.mlp."
    h = torch.relu(_linear(x, sd[p + "fc.weight"]))
    return _linear(h.square(), sd[p + "proj.weight"])


def _block_mix(sd: dict[str, torch.Tensor], block_idx: int, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    mix = sd[f"blocks.{block_idx}.resid_mix"].to(dtype=x.dtype)
    return mix[0][None, None, :] * x + mix[1][None, None, :] * x0


def _block_forward_reference(
    sd: dict[str, torch.Tensor],
    block_idx: int,
    x: torch.Tensor,
    x0: torch.Tensor,
    arch: dict[str, int | bool],
    lower_bound: float | None,
    allow_neg_eigval: bool,
    l2norm_eps: float,
    o_norm_eps: float,
) -> torch.Tensor:
    x = _block_mix(sd, block_idx, x, x0)
    attn_in = _rms_norm_last_dim(x, eps=1e-6)
    attn_out, _ = _kda_forward_reference(
        sd=sd,
        block_idx=block_idx,
        hidden_states=attn_in,
        num_heads=int(arch["num_heads"]),
        head_k_dim=int(arch["head_k_dim"]),
        head_v_dim=int(arch["head_v_dim"]),
        use_short_conv=bool(arch["use_short_conv"]),
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        l2norm_eps=l2norm_eps,
        o_norm_eps=o_norm_eps,
    )
    x = x + sd[f"blocks.{block_idx}.attn_scale"].to(dtype=x.dtype)[None, None, :] * attn_out
    x = x + sd[f"blocks.{block_idx}.mlp_scale"].to(dtype=x.dtype)[None, None, :] * _mlp_forward_reference(
        sd,
        block_idx,
        _rms_norm_last_dim(x, eps=1e-6),
    )
    return x


def _extract_kda_tensors_for_layer(
    sd: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    layer_idx: int,
    arch: dict[str, int | bool],
    lower_bound: float | None,
    allow_neg_eigval: bool,
    l2norm_eps: float,
    o_norm_eps: float,
) -> dict[str, torch.Tensor]:
    num_layers = int(arch["num_layers"])
    num_encoder_layers = int(arch["num_encoder_layers"])
    num_decoder_layers = int(arch["num_decoder_layers"])
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx must be in [0, {num_layers - 1}]")

    x = F.embedding(input_ids, sd["tok_emb.weight"]).to(dtype=sd["tok_emb.weight"].dtype)
    x = _rms_norm_last_dim(x, eps=1e-6)
    x0 = x
    skips: list[torch.Tensor] = []

    for i in range(num_encoder_layers):
        if i == layer_idx:
            xmix = _block_mix(sd, i, x, x0)
            attn_in = _rms_norm_last_dim(xmix, eps=1e-6)
            _, ctx = _kda_forward_reference(
                sd=sd,
                block_idx=i,
                hidden_states=attn_in,
                num_heads=int(arch["num_heads"]),
                head_k_dim=int(arch["head_k_dim"]),
                head_v_dim=int(arch["head_v_dim"]),
                use_short_conv=bool(arch["use_short_conv"]),
                lower_bound=lower_bound,
                allow_neg_eigval=allow_neg_eigval,
                l2norm_eps=l2norm_eps,
                o_norm_eps=o_norm_eps,
            )
            return ctx
        x = _block_forward_reference(sd, i, x, x0, arch, lower_bound, allow_neg_eigval, l2norm_eps, o_norm_eps)
        skips.append(x)

    for j in range(num_decoder_layers):
        if skips and int(sd["skip_weights"].shape[0]) > 0:
            x = x + sd["skip_weights"][j].to(dtype=x.dtype)[None, None, :] * skips.pop()
        idx = num_encoder_layers + j
        if idx == layer_idx:
            xmix = _block_mix(sd, idx, x, x0)
            attn_in = _rms_norm_last_dim(xmix, eps=1e-6)
            _, ctx = _kda_forward_reference(
                sd=sd,
                block_idx=idx,
                hidden_states=attn_in,
                num_heads=int(arch["num_heads"]),
                head_k_dim=int(arch["head_k_dim"]),
                head_v_dim=int(arch["head_v_dim"]),
                use_short_conv=bool(arch["use_short_conv"]),
                lower_bound=lower_bound,
                allow_neg_eigval=allow_neg_eigval,
                l2norm_eps=l2norm_eps,
                o_norm_eps=o_norm_eps,
            )
            return ctx
        x = _block_forward_reference(sd, idx, x, x0, arch, lower_bound, allow_neg_eigval, l2norm_eps, o_norm_eps)

    raise RuntimeError("Failed to extract KDA tensors for requested layer")


def _load_data_shard(path: Path) -> torch.Tensor:
    if np is None:
        raise RuntimeError("NumPy is required for --token-mode shard")
    header = np.fromfile(path, dtype="<i4", count=256)
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size} bytes")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


def _make_input_ids(
    mode: str,
    batch: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    device: str,
    token_shard: Path | None,
    token_offset: int,
) -> torch.Tensor:
    if mode == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        ids = torch.randint(0, vocab_size, (batch, seq_len), device="cpu", dtype=torch.long, generator=gen)
        return ids.to(device)
    if mode == "shard":
        if token_shard is None:
            raise ValueError("--token-shard is required when --token-mode=shard")
        tokens = _load_data_shard(token_shard)
        need = batch * seq_len
        if token_offset < 0 or token_offset + need > tokens.numel():
            raise ValueError(
                f"Requested token range [{token_offset}, {token_offset + need}) exceeds shard length {tokens.numel()}"
            )
        ids = tokens[token_offset: token_offset + need].to(dtype=torch.long)
        return ids.reshape(batch, seq_len).to(device)
    raise ValueError(f"Unsupported token mode: {mode}")


def _step_affine(
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decay = torch.exp(g_t)
    m_t = torch.diag_embed(decay) - beta_t[:, None, None] * k_t[:, :, None] * (k_t * decay)[:, None, :]
    b_t = beta_t[:, None, None] * k_t[:, :, None] * v_t[:, None, :]
    return m_t, b_t


def _chunk_affine_chain(
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, tlen, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    bh = bsz * heads
    n_chunks = (tlen + chunk_size - 1) // chunk_size

    eye = torch.eye(key_dim, device=k.device, dtype=k.dtype).unsqueeze(0).expand(bh, -1, -1)
    zeros = torch.zeros(bh, key_dim, value_dim, device=k.device, dtype=k.dtype)

    ms = []
    bs = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(tlen, start + chunk_size)
        m_chunk = eye.clone()
        b_chunk = zeros.clone()

        for t in range(start, end):
            k_t = k[:, t].reshape(bh, key_dim)
            v_t = v[:, t].reshape(bh, value_dim)
            g_t = gk[:, t].reshape(bh, key_dim)
            beta_t = beta[:, t].reshape(bh)
            m_t, b_t = _step_affine(k_t, v_t, g_t, beta_t)
            m_chunk = torch.matmul(m_t, m_chunk)
            b_chunk = torch.matmul(m_t, b_chunk) + b_t

        ms.append(m_chunk)
        bs.append(b_chunk)

    return torch.stack(ms, dim=0), torch.stack(bs, dim=0)


def _chunk_ut_token_matrices(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute UT-form per-chunk token matrix M_[t] of shape [N_chunk, BH, C, C].

    This corresponds to Eq. (6)/(7) style intra-chunk matrix used to map V -> U.
    """
    bsz, tlen, heads, key_dim = k.shape
    bh = bsz * heads
    if tlen % chunk_size != 0:
        raise ValueError(
            f"UT token matrix probe requires seq_len divisible by chunk_size, got {tlen} and {chunk_size}"
        )
    n_chunks = tlen // chunk_size
    c = chunk_size

    eye = torch.eye(c, device=k.device, dtype=k.dtype)
    strict_upper_mask = torch.triu(torch.ones(c, c, dtype=torch.bool, device=k.device), diagonal=0)

    out = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * c
        end = start + c
        # [BH, C, K], [BH, C, K], [BH, C]
        k_c = k[:, start:end].permute(0, 2, 1, 3).reshape(bh, c, key_dim)
        g_c = g[:, start:end].permute(0, 2, 1, 3).reshape(bh, c, key_dim)
        beta_c = beta[:, start:end].permute(0, 2, 1).reshape(bh, c)

        # Cumulative log-decay within chunk.
        g_cum = torch.cumsum(g_c, dim=1)

        # Build lower-triangular solve matrix using the same recurrence as naive_chunk_kda.
        a = torch.zeros(bh, c, c, dtype=k.dtype, device=k.device)
        for i in range(c):
            k_i = k_c[:, i, :]          # [BH, K]
            g_i = g_cum[:, i:i + 1, :]  # [BH, 1, K]
            a[:, i, :] = torch.einsum("b c d, b d -> b c", k_c * torch.exp(g_cum - g_i), k_i)
        a = a * beta_c[:, :, None]
        a = -a.masked_fill(strict_upper_mask[None], 0)
        for i in range(1, c):
            a[:, i, :i] = a[:, i, :i] + (a[:, i, :, None] * a[:, :, :i]).sum(-2)
        m = (a + eye[None]) * beta_c[:, None, :]
        out.append(m)

    return torch.stack(out, dim=0)


def _local_chunk_cumsum_scaled(g: torch.Tensor, chunk_size: int, scale: float) -> torch.Tensor:
    bsz, tlen, heads, dim = g.shape
    out = torch.empty_like(g)
    for start in range(0, tlen, chunk_size):
        end = min(tlen, start + chunk_size)
        out[:, start:end] = torch.cumsum(g[:, start:end], dim=1) * scale
    return out


def _chunk_ut_token_matrices_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError("matrix-kind=ut_token_kernel currently requires CUDA.")
    if q.shape[1] % chunk_size != 0:
        raise ValueError(
            f"ut_token_kernel requires seq_len divisible by chunk_size, got {q.shape[1]} and {chunk_size}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    vendored_fla = repo_root / "experimental" / "fla_src"
    vendored_fla_str = str(vendored_fla)
    if vendored_fla.is_dir() and vendored_fla_str not in sys.path:
        sys.path.insert(0, vendored_fla_str)

    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

    # chunk_kda_fwd_intra expects chunk-local cumsum in log2 space.
    gk = _local_chunk_cumsum_scaled(g, chunk_size=chunk_size, scale=1.0 / math.log(2.0))
    scale = q.shape[-1] ** -0.5
    with torch.inference_mode():
        _, _, _, _, _, akk = chunk_kda_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=gk,
            beta=beta,
            scale=scale,
            chunk_size=chunk_size,
            safe_gate=False,
            disable_recompute=True,
        )
    # akk: [B, T, H, C] where rows are token index inside chunk, cols are chunk columns.
    bsz, tlen, heads, c = akk.shape
    n_chunks = tlen // c
    return akk.view(bsz, n_chunks, c, heads, c).permute(1, 0, 3, 2, 4).reshape(n_chunks, bsz * heads, c, c)


def _compose_chunks(
    m_chunks: torch.Tensor,
    b_chunks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_chunks, bh, key_dim, _ = m_chunks.shape
    value_dim = b_chunks.shape[-1]
    m_total = torch.eye(key_dim, device=m_chunks.device, dtype=m_chunks.dtype).unsqueeze(0).expand(bh, -1, -1).clone()
    b_total = torch.zeros(bh, key_dim, value_dim, device=m_chunks.device, dtype=m_chunks.dtype)
    for i in range(n_chunks):
        m_total = torch.matmul(m_chunks[i], m_total)
        b_total = torch.matmul(m_chunks[i], b_total) + b_chunks[i]
    return m_total, b_total


def _distance_profile(m: torch.Tensor) -> list[float]:
    abs_m = m.abs()
    key_dim = abs_m.shape[-1]
    idx = torch.arange(key_dim, device=abs_m.device)
    dist = (idx[:, None] - idx[None, :]).abs()
    profile = []
    for d in range(key_dim):
        vals = abs_m[:, dist == d]
        profile.append(float(vals.mean().item()))
    return profile


def _lag_profile_lower(m: torch.Tensor) -> list[float]:
    """Mean |M_ij| by causal lag l=i-j for lower-triangular token matrix."""
    abs_m = m.abs()
    c = abs_m.shape[-1]
    idx = torch.arange(c, device=abs_m.device)
    lag = idx[:, None] - idx[None, :]
    profile = []
    for l in range(c):
        vals = abs_m[:, lag == l]
        profile.append(float(vals.mean().item()))
    return profile


def _count_monotonicity_violations(profile: list[float], tol: float = 0.0) -> int:
    violations = 0
    for i in range(len(profile) - 1):
        if profile[i + 1] > profile[i] + tol:
            violations += 1
    return violations


def run(args: argparse.Namespace) -> None:
    if args.fast_smoke:
        args.batch = 1
        args.seq_len = 64
        args.chunk_size = 32
        args.max_dist_to_print = 8
        args.preview_size = 4
        args.layer_idx = 0 if args.layer_idx is None else args.layer_idx

    device = _device_from_name(args.device)
    dtype = _dtype_from_name(args.dtype)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()

    sd_cpu = _load_state_dict(ckpt_path)
    arch = _infer_arch(sd_cpu)
    sd = _cast_state(sd_cpu, device=device, dtype=dtype)

    layer_idx = args.layer_idx if args.layer_idx is not None else 0
    input_ids = _make_input_ids(
        mode=args.token_mode,
        batch=args.batch,
        seq_len=args.seq_len,
        vocab_size=int(arch["vocab_size"]),
        seed=args.seed,
        device=device,
        token_shard=Path(args.token_shard).expanduser().resolve() if args.token_shard else None,
        token_offset=args.token_offset,
    )

    with torch.inference_mode():
        ctx = _extract_kda_tensors_for_layer(
            sd=sd,
            input_ids=input_ids,
            layer_idx=layer_idx,
            arch=arch,
            lower_bound=args.lower_bound,
            allow_neg_eigval=args.allow_neg_eigval,
            l2norm_eps=args.l2norm_eps,
            o_norm_eps=args.o_norm_eps,
        )

    k = ctx["k"]
    q = ctx["q"]
    g = ctx["g"]
    beta = ctx["beta"]
    v = ctx["v"]

    bh = args.batch * int(arch["num_heads"])
    key_dim = int(arch["head_k_dim"])
    value_dim = int(arch["head_v_dim"])

    print("KDA M-matrix probe (trained checkpoint)")
    print(f"checkpoint={ckpt_path}")
    print(
        f"device={device} dtype={dtype} seed={args.seed} token_mode={args.token_mode} "
        f"layer_idx={layer_idx} B={args.batch} T={args.seq_len} "
        f"H={arch['num_heads']} K={arch['head_k_dim']} V={arch['head_v_dim']} chunk={args.chunk_size}"
    )
    print(
        f"use_short_conv={arch['use_short_conv']} "
        f"lower_bound={args.lower_bound} allow_neg_eigval={args.allow_neg_eigval}"
    )
    print(f"matrix_kind={args.matrix_kind}")
    print("")

    if args.matrix_kind == "state":
        m_chunks, b_chunks = _chunk_affine_chain(k, v, g, beta, args.chunk_size)
        m_total, b_total = _compose_chunks(m_chunks, b_chunks)

        h0 = args.init_state_scale * torch.randn(bh, key_dim, value_dim, device=device, dtype=dtype)
        s_pred = torch.matmul(m_total, h0) + b_total

        q_dummy = torch.zeros_like(k)
        _, s_ref = _naive_recurrent_kda(
            q=q_dummy,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0.view(args.batch, int(arch["num_heads"]), key_dim, value_dim),
            output_final_state=True,
            compute_dtype=dtype,
        )
        s_ref = s_ref.view(bh, key_dim, value_dim)
        max_err = float((s_ref - s_pred).abs().max().item())
        mean_err = float((s_ref - s_pred).abs().mean().item())
        print(f"state consistency (affine M vs recurrent): max_err={max_err:.3e} mean_err={mean_err:.3e}")
        print("")

        chunk_profile = _distance_profile(m_chunks.reshape(-1, key_dim, key_dim))
        full_profile = _distance_profile(m_total.reshape(-1, key_dim, key_dim))
        chunk_violations = _count_monotonicity_violations(chunk_profile, tol=args.monotonic_tol)
        full_violations = _count_monotonicity_violations(full_profile, tol=args.monotonic_tol)

        max_dist = min(args.max_dist_to_print, key_dim)
        print("Per-chunk state-transition M |M_ij| mean by feature distance d=|i-j|")
        print("d\tmean_abs\tratio_to_diag")
        diag0 = chunk_profile[0]
        for d in range(max_dist):
            ratio = chunk_profile[d] / diag0 if diag0 > 0 else float("nan")
            print(f"{d}\t{chunk_profile[d]:.6e}\t{ratio:.6f}")
        print(f"monotonic violations (tol={args.monotonic_tol}): {chunk_violations}/{key_dim - 1}")
        print("")

        print("Full-sequence state-transition M_total |M_ij| mean by feature distance d=|i-j|")
        print("d\tmean_abs\tratio_to_diag")
        diag0 = full_profile[0]
        for d in range(max_dist):
            ratio = full_profile[d] / diag0 if diag0 > 0 else float("nan")
            print(f"{d}\t{full_profile[d]:.6e}\t{ratio:.6f}")
        print(f"monotonic violations (tol={args.monotonic_tol}): {full_violations}/{key_dim - 1}")
        print("")

        if args.preview_size > 0:
            preview = min(args.preview_size, key_dim)
            torch.set_printoptions(precision=4, sci_mode=True, linewidth=180)
            first_chunk_first_head = m_chunks[0, 0, :preview, :preview].detach().cpu()
            print(f"Preview first chunk state-transition M[0,0], top-left {preview}x{preview}:")
            print(first_chunk_first_head)
        return

    # UT token matrix path (Eq. 6/7 style): M_[t] in R^{C x C}
    c = args.chunk_size
    if args.matrix_kind == "ut_token":
        m_ut = _chunk_ut_token_matrices(k, g, beta, c)  # [N_chunk, BH, C, C]
    else:
        m_ut = _chunk_ut_token_matrices_kernel(q, k, v, g, beta, c)
    ut_profile = _lag_profile_lower(m_ut.reshape(-1, c, c))
    ut_violations = _count_monotonicity_violations(ut_profile, tol=args.monotonic_tol)

    max_dist = min(args.max_dist_to_print, c)
    print("Per-chunk UT M_[t] |M_ij| mean by token lag l=i-j (lower-triangular)")
    print("l\tmean_abs\tratio_to_diag")
    diag0 = ut_profile[0]
    for l in range(max_dist):
        ratio = ut_profile[l] / diag0 if diag0 > 0 else float("nan")
        print(f"{l}\t{ut_profile[l]:.6e}\t{ratio:.6f}")
    print(f"monotonic violations (tol={args.monotonic_tol}): {ut_violations}/{c - 1}")
    print("")

    if args.preview_size > 0:
        preview = min(args.preview_size, c)
        torch.set_printoptions(precision=4, sci_mode=True, linewidth=180)
        first_chunk_first_head = m_ut[0, 0, :preview, :preview].detach().cpu()
        print(f"Preview first chunk UT M_[0][0], top-left {preview}x{preview}:")
        print(first_chunk_first_head)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe KDA transition matrix M from a trained checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to final_model.pt-style state_dict checkpoint.")
    p.add_argument("--fast-smoke", action="store_true", help="Small quick preset for laptop CPU.")
    p.add_argument("--layer-idx", type=int, default=0, help="Block index to inspect (default: 0).")

    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--dtype", type=str, default="float32", help="float32|float64")

    p.add_argument("--token-mode", type=str, default="random", choices=["random", "shard"])
    p.add_argument("--token-shard", type=str, default=None, help="Path to fineweb_*.bin shard when token-mode=shard.")
    p.add_argument("--token-offset", type=int, default=0, help="Offset into token shard for token-mode=shard.")

    p.add_argument("--lower-bound", type=float, default=-5.0)
    p.add_argument("--allow-neg-eigval", action="store_true")
    p.add_argument("--l2norm-eps", type=float, default=1e-6)
    p.add_argument("--o-norm-eps", type=float, default=1e-5)
    p.add_argument("--init-state-scale", type=float, default=1.0)
    p.add_argument(
        "--matrix-kind",
        type=str,
        default="state",
        choices=["state", "ut_token", "ut_token_kernel"],
        help=(
            "`state`: d_k x d_k state-transition M. "
            "`ut_token`: C x C UT M_[t] from torch reference reconstruction. "
            "`ut_token_kernel`: C x C UT M_[t] read from intra-chunk kernel output (CUDA)."
        ),
    )

    p.add_argument("--max-dist-to-print", type=int, default=16)
    p.add_argument("--preview-size", type=int, default=8)
    p.add_argument("--monotonic-tol", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
