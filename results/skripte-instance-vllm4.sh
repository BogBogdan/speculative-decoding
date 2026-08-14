#!/bin/bash
export PATH="$HOME/venv-vllm/bin:$PATH"
export CUDA_HOME=/usr
cd ~/speculative-decoding
export DRAFT_ID=/home/ubuntu/draft-151665
export TARGET_ID=/home/ubuntu/target-151665
export MAX_NOVIH=128 BROJ_PREFIKSA=20 GPU_UTIL=0.85
for G in 0 1 3 5 8; do
  echo "######## GAMMA=$G ########"
  GAMMA=$G ~/venv-vllm/bin/python scripts/run_eval_vllm.py 2>&1 \
    | grep -E "baseline \(|spekulativno,|vreme |tokena |propusnost|alfa |Error|error:|RuntimeError|FileNotFound" | head -10
  sleep 5
done
echo GOTOVO_VLLM4
