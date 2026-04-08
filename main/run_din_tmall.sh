#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU="${GPU:--1}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
BATCH_SIZE="${BATCH_SIZE:-256}"

python run_expid.py \
  --config ./config/General_config \
  --expid Din_tmall \
  --dataset tmall_900 \
  --gpu "${GPU}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --batch_size "${BATCH_SIZE}"
