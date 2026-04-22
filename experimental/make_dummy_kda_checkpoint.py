#!/usr/bin/env python3
from __future__ import annotations

"""Create a minimal KDA-compatible checkpoint for local probing.

This is useful on Macs/CPU-only machines where `train_gpt.py` (CUDA-only) cannot
produce `final_model.pt`. The output checkpoint matches the key/shape layout
expected by `experimental/kda_m_matrix_from_checkpoint.py`.
"""

import argparse
from pathlib import Path

import torch


def build_state_dict(
    *,
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    mlp_mult: int,
    conv_size: int,
    use_short_conv: bool,
    seed: int,
    std: float,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return std * torch.randn(*shape, generator=gen, dtype=torch.float32)

    sd: dict[str, torch.Tensor] = {}

    key_dim = num_heads * head_k_dim
    value_dim = num_heads * head_v_dim
    hidden = mlp_mult * model_dim
    num_encoder_layers = num_layers // 2
    num_decoder_layers = num_layers - num_encoder_layers
    num_skip_weights = min(num_encoder_layers, num_decoder_layers)

    sd["tok_emb.weight"] = randn(vocab_size, model_dim)
    sd["skip_weights"] = torch.ones(num_skip_weights, model_dim, dtype=torch.float32)

    for i in range(num_layers):
        p = f"blocks.{i}."
        sd[p + "resid_mix"] = torch.stack(
            (
                torch.ones(model_dim, dtype=torch.float32),
                torch.zeros(model_dim, dtype=torch.float32),
            ),
            dim=0,
        )
        sd[p + "attn_scale"] = torch.ones(model_dim, dtype=torch.float32)
        sd[p + "mlp_scale"] = torch.ones(model_dim, dtype=torch.float32)

        # MLP
        sd[p + "mlp.fc.weight"] = randn(hidden, model_dim)
        sd[p + "mlp.proj.weight"] = randn(model_dim, hidden)

        # KDA projections
        kdap = p + "attn.kda."
        sd[kdap + "q_proj.weight"] = randn(key_dim, model_dim)
        sd[kdap + "k_proj.weight"] = randn(key_dim, model_dim)
        sd[kdap + "v_proj.weight"] = randn(value_dim, model_dim)

        if use_short_conv:
            sd[kdap + "q_conv1d.weight"] = randn(key_dim, 1, conv_size)
            sd[kdap + "k_conv1d.weight"] = randn(key_dim, 1, conv_size)
            sd[kdap + "v_conv1d.weight"] = randn(value_dim, 1, conv_size)

        sd[kdap + "f_proj.0.weight"] = randn(head_v_dim, model_dim)
        sd[kdap + "f_proj.1.weight"] = randn(key_dim, head_v_dim)
        sd[kdap + "b_proj.weight"] = randn(num_heads, model_dim)

        # Control params
        sd[kdap + "A_log"] = torch.zeros(num_heads, dtype=torch.float32)
        sd[kdap + "dt_bias"] = randn(key_dim)

        # Output gate/norm/proj
        sd[kdap + "g_proj.0.weight"] = randn(head_v_dim, model_dim)
        sd[kdap + "g_proj.1.weight"] = randn(value_dim, head_v_dim)
        sd[kdap + "g_proj.1.bias"] = randn(value_dim)
        sd[kdap + "o_norm.weight"] = torch.ones(head_v_dim, dtype=torch.float32)
        sd[kdap + "o_proj.weight"] = randn(model_dim, value_dim)

    return sd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a dummy KDA checkpoint for local probing.")
    p.add_argument("--out", type=str, default="dummy_kda_checkpoint.pt")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=9)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-k-dim", type=int, default=64)
    p.add_argument("--head-v-dim", type=int, default=64)
    p.add_argument("--mlp-mult", type=int, default=2)
    p.add_argument("--conv-size", type=int, default=4)
    p.add_argument("--no-short-conv", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--std", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sd = build_state_dict(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
        mlp_mult=args.mlp_mult,
        conv_size=args.conv_size,
        use_short_conv=not args.no_short_conv,
        seed=args.seed,
        std=args.std,
    )
    torch.save(sd, out)
    print(f"saved checkpoint: {out}")
    print(f"tensors: {len(sd)}")


if __name__ == "__main__":
    main()
