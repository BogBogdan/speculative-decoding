#!/bin/bash
# Baseline pa spekulativno za vise gama, svaki u svom procesu.
# vLLM zauzme KV cache po instanci, pa se ne smeju praviti u istom procesu.
#
#   bash scripts/run_vllm.sh 2>&1 | tee vllm_rezultati.log

set -u
PY=${PY:-python}
export TARGET_ID=${TARGET_ID:-Qwen/Qwen2.5-14B}
export DRAFT_ID=${DRAFT_ID:-Qwen/Qwen2.5-0.5B}
export MAX_NOVIH=${MAX_NOVIH:-128}
export BROJ_PREFIKSA=${BROJ_PREFIKSA:-3}
export GPU_UTIL=${GPU_UTIL:-0.85}

for G in 0 1 3 5 8; do
  echo
  echo "############ GAMMA=$G ############"
  GAMMA=$G $PY scripts/run_eval_vllm.py || echo "GAMMA=$G pao"
  sleep 5          # da se GPU memorija oslobodi pre sledece instance
done
echo
echo "GOTOVO"
