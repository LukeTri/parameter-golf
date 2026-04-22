# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Lightweight package init for vendored FLA.

The upstream `fla.__init__` eagerly imports the full layer/model registry,
which pulls optional dependencies (for example `transformers`) that are not
required for `train_gpt.py` KDA usage.

We keep this init minimal so `from fla.layers.kda import KimiDeltaAttention`
works in lean training environments.
"""

__version__ = "0.5.0"

