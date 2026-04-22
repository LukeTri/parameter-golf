# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Lightweight layer exports for vendored FLA.

`train_gpt.py` only needs KDA. Keeping this package init minimal avoids
importing unrelated layers that may require optional dependencies.
"""

from .kda import KimiDeltaAttention

__all__ = [
    "KimiDeltaAttention",
]

