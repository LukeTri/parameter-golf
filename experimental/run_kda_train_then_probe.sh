#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# -----------------------------
# Tunables (override via env)
# -----------------------------
: "${PYTHON:=python3}"
: "${NPROC_PER_NODE:=1}"
: "${RUN_ID:=kda_probe_$(date +%Y%m%d_%H%M%S)}"

# Dataset/tokenizer
: "${DATASET_VARIANT:=sp1024}"
: "${TRAIN_SHARDS:=1}"
: "${DATA_PATH:=${ROOT_DIR}/data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=${ROOT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"

# Train config (safe defaults for a single consumer GPU)
: "${ITERATIONS:=200}"
: "${MAX_WALLCLOCK_SECONDS:=300}"
: "${TRAIN_SEQ_LEN:=512}"
: "${TRAIN_BATCH_TOKENS:=131072}"
: "${VAL_BATCH_SIZE:=32768}"
: "${VAL_LOSS_EVERY:=0}"
: "${TRAIN_LOG_EVERY:=20}"
: "${SEED:=1337}"

# KDA config
: "${ATTN_IMPL:=kda}"
: "${NUM_LAYERS:=9}"
: "${MODEL_DIM:=512}"
: "${NUM_HEADS:=8}"
: "${NUM_KV_HEADS:=${NUM_HEADS}}"
: "${MLP_MULT:=2}"
: "${KDA_MODE:=chunk}"
: "${KDA_USE_SHORT_CONV:=1}"
: "${KDA_ALLOW_NEG_EIGVAL:=0}"
: "${KDA_COMPILE_MODEL:=0}"

# Probe config
: "${PROBE_LAYER_IDX:=0}"
: "${PROBE_BATCH:=1}"
: "${PROBE_SEQ_LEN:=256}"
: "${PROBE_CHUNK_SIZE:=64}"
: "${PROBE_DEVICE:=auto}"
: "${PROBE_DTYPE:=float32}"
: "${PROBE_MATRIX_KIND:=ut_token}"
: "${PROBE_TOKEN_OFFSET:=0}"
: "${PROBE_LOWER_BOUND:=-5.0}"

echo "==> Preflight: KDA import check"
if ! "${PYTHON}" - <<'PY'
import sys
from pathlib import Path
root = Path(".").resolve()
vendored = root / "experimental" / "fla_src"
if str(vendored) not in sys.path:
    sys.path.insert(0, str(vendored))
from fla.layers.kda import KimiDeltaAttention  # noqa: F401
print("KDA import OK")
PY
then
  echo "KDA import failed; installing fallback deps (einops, transformers) ..."
  "${PYTHON}" -m pip install einops transformers
  "${PYTHON}" - <<'PY'
import sys
from pathlib import Path
root = Path(".").resolve()
vendored = root / "experimental" / "fla_src"
if str(vendored) not in sys.path:
    sys.path.insert(0, str(vendored))
from fla.layers.kda import KimiDeltaAttention  # noqa: F401
print("KDA import OK after fallback install")
PY
fi

echo "==> Checking dataset shards..."
if ls "${DATA_PATH}"/fineweb_train_*.bin >/dev/null 2>&1; then
  echo "Found dataset at ${DATA_PATH}"
else
  echo "Dataset not found. Downloading variant=${DATASET_VARIANT} train_shards=${TRAIN_SHARDS} ..."
  "${PYTHON}" data/cached_challenge_fineweb.py --variant "${DATASET_VARIANT}" --train-shards "${TRAIN_SHARDS}"
fi

echo "==> Training KDA model (RUN_ID=${RUN_ID})"
RUN_ID="${RUN_ID}" \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
VOCAB_SIZE="${VOCAB_SIZE}" \
SEED="${SEED}" \
ITERATIONS="${ITERATIONS}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" \
ATTN_IMPL="${ATTN_IMPL}" \
NUM_LAYERS="${NUM_LAYERS}" \
MODEL_DIM="${MODEL_DIM}" \
NUM_HEADS="${NUM_HEADS}" \
NUM_KV_HEADS="${NUM_KV_HEADS}" \
MLP_MULT="${MLP_MULT}" \
KDA_MODE="${KDA_MODE}" \
KDA_USE_SHORT_CONV="${KDA_USE_SHORT_CONV}" \
KDA_ALLOW_NEG_EIGVAL="${KDA_ALLOW_NEG_EIGVAL}" \
KDA_COMPILE_MODEL="${KDA_COMPILE_MODEL}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py

if [[ ! -f "${ROOT_DIR}/final_model.pt" ]]; then
  echo "ERROR: final_model.pt not found after training."
  exit 1
fi

echo "==> Running M-matrix probe from trained checkpoint"
VAL_SHARD="$(ls "${DATA_PATH}"/fineweb_val_*.bin 2>/dev/null | head -n 1 || true)"
PROBE_ARGS=(
  --checkpoint "${ROOT_DIR}/final_model.pt"
  --layer-idx "${PROBE_LAYER_IDX}"
  --batch "${PROBE_BATCH}"
  --seq-len "${PROBE_SEQ_LEN}"
  --chunk-size "${PROBE_CHUNK_SIZE}"
  --device "${PROBE_DEVICE}"
  --dtype "${PROBE_DTYPE}"
  --matrix-kind "${PROBE_MATRIX_KIND}"
  --lower-bound "${PROBE_LOWER_BOUND}"
  --token-offset "${PROBE_TOKEN_OFFSET}"
)
if [[ "${KDA_ALLOW_NEG_EIGVAL}" == "1" ]]; then
  PROBE_ARGS+=(--allow-neg-eigval)
fi
if [[ -n "${VAL_SHARD}" ]]; then
  PROBE_ARGS+=(--token-mode shard --token-shard "${VAL_SHARD}")
else
  PROBE_ARGS+=(--token-mode random)
fi

"${PYTHON}" experimental/kda_m_matrix_from_checkpoint.py "${PROBE_ARGS[@]}"

echo "==> Done."
