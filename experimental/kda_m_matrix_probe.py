#!/usr/bin/env python3
from __future__ import annotations

"""Probe the KDA state transition matrix M and its diagonal-distance profile.

For each token, this uses the standard recurrent KDA update:
M_t = diag(exp(g_t)) - beta_t * k_t @ (k_t * exp(g_t))^T
Then composes token transitions into per-chunk and full-sequence M.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FLA_SRC = REPO_ROOT / "experimental" / "fla_src"
if str(FLA_SRC) not in sys.path:
    sys.path.insert(0, str(FLA_SRC))

NAIVE_IMPL_NAME = "local_fallback"
try:
    from fla.ops.kda.naive import naive_recurrent_kda as _naive_recurrent_kda  # noqa: E402
    NAIVE_IMPL_NAME = "fla.ops.kda.naive.naive_recurrent_kda"
except Exception:
    # Fallback keeps this script runnable when optional FLA deps are missing.
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

        s = k.new_zeros(bsz, heads, key_dim, value_dim).to(q)
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


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _device_from_name(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _make_inputs(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    max_log_decay: float,
    beta_max: float,
    init_state_scale: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    q = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=dtype, generator=g)
    k = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=dtype, generator=g)
    v = torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=dtype, generator=g)

    # beta in [0, beta_max]
    beta = beta_max * torch.rand(batch, seq_len, heads, device=device, dtype=dtype, generator=g)

    # log-space decay <= 0, so exp(g) is in [exp(-max_log_decay), 1]
    gk = -max_log_decay * torch.rand(batch, seq_len, heads, key_dim, device=device, dtype=dtype, generator=g)

    h0 = init_state_scale * torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=dtype, generator=g)
    return q, k, v, gk, beta, h0


def _step_affine(
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # k_t, g_t: [BH, K], v_t: [BH, V], beta_t: [BH]
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
    # m: [N, K, K]
    abs_m = m.abs()
    key_dim = abs_m.shape[-1]
    idx = torch.arange(key_dim, device=abs_m.device)
    dist = (idx[:, None] - idx[None, :]).abs()
    profile = []
    for d in range(key_dim):
        vals = abs_m[:, dist == d]
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
        args.heads = 2
        args.key_dim = 16
        args.value_dim = 16
        args.chunk_size = 32
        args.max_dist_to_print = 8
        args.preview_size = 4

    device = _device_from_name(args.device)
    dtype = _dtype_from_name(args.dtype)

    q, k, v, gk, beta, h0 = _make_inputs(
        batch=args.batch,
        seq_len=args.seq_len,
        heads=args.heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        max_log_decay=args.max_log_decay,
        beta_max=args.beta_max,
        init_state_scale=args.init_state_scale,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    m_chunks, b_chunks = _chunk_affine_chain(k, v, gk, beta, args.chunk_size)
    m_total, b_total = _compose_chunks(m_chunks, b_chunks)

    bh = args.batch * args.heads
    h0_flat = h0.reshape(bh, args.key_dim, args.value_dim)
    s_pred = torch.matmul(m_total, h0_flat) + b_total

    # Compare with the standard reference recurrence.
    _, s_ref = _naive_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=gk,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        compute_dtype=dtype,
    )
    s_ref = s_ref.reshape(bh, args.key_dim, args.value_dim)
    max_err = float((s_ref - s_pred).abs().max().item())
    mean_err = float((s_ref - s_pred).abs().mean().item())

    chunk_profile = _distance_profile(m_chunks.reshape(-1, args.key_dim, args.key_dim))
    full_profile = _distance_profile(m_total.reshape(-1, args.key_dim, args.key_dim))
    chunk_violations = _count_monotonicity_violations(chunk_profile, tol=args.monotonic_tol)
    full_violations = _count_monotonicity_violations(full_profile, tol=args.monotonic_tol)

    print("KDA M-matrix probe")
    print(
        f"device={device} dtype={dtype} seed={args.seed} "
        f"B={args.batch} T={args.seq_len} H={args.heads} K={args.key_dim} V={args.value_dim} chunk={args.chunk_size}"
    )
    print(f"naive_impl={NAIVE_IMPL_NAME}")
    print(f"max_log_decay={args.max_log_decay:.3f} beta_max={args.beta_max:.3f}")
    print(f"state consistency vs naive recurrence: max_err={max_err:.3e} mean_err={mean_err:.3e}")
    print("")

    max_dist = min(args.max_dist_to_print, args.key_dim)
    print("Per-chunk M |M_ij| mean by distance d=|i-j|")
    print("d\tmean_abs\tratio_to_diag")
    diag0 = chunk_profile[0]
    for d in range(max_dist):
        ratio = chunk_profile[d] / diag0 if diag0 > 0 else float("nan")
        print(f"{d}\t{chunk_profile[d]:.6e}\t{ratio:.6f}")
    print(f"monotonic violations (tol={args.monotonic_tol}): {chunk_violations}/{args.key_dim - 1}")
    print("")

    print("Full-sequence M_total |M_ij| mean by distance d=|i-j|")
    print("d\tmean_abs\tratio_to_diag")
    diag0 = full_profile[0]
    for d in range(max_dist):
        ratio = full_profile[d] / diag0 if diag0 > 0 else float("nan")
        print(f"{d}\t{full_profile[d]:.6e}\t{ratio:.6f}")
    print(f"monotonic violations (tol={args.monotonic_tol}): {full_violations}/{args.key_dim - 1}")
    print("")

    if args.preview_size > 0:
        preview = min(args.preview_size, args.key_dim)
        torch.set_printoptions(precision=4, sci_mode=True, linewidth=180)
        first_chunk_first_head = m_chunks[0, 0, :preview, :preview].detach().cpu()
        print(f"Preview first chunk M[0,0], top-left {preview}x{preview}:")
        print(first_chunk_first_head)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe KDA transition matrix M and diagonal-distance decay.")
    p.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Run a small quick preset suitable for laptop CPU.",
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--dtype", type=str, default="float32", help="float32|float64")
    p.add_argument("--max-log-decay", type=float, default=3.0)
    p.add_argument("--beta-max", type=float, default=1.0)
    p.add_argument("--init-state-scale", type=float, default=1.0)
    p.add_argument("--max-dist-to-print", type=int, default=16)
    p.add_argument("--preview-size", type=int, default=8)
    p.add_argument("--monotonic-tol", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
