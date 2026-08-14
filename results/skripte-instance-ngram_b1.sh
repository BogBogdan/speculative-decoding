#!/bin/bash
export PATH="$HOME/venv-vllm/bin:$PATH" CUDA_HOME=/usr
cd ~/speculative-decoding
export DRAFT_ID=$HOME/draft-151665 TARGET_ID=$HOME/target-151665
export MAX_NOVIH=128 BROJ_PREFIKSA=20 GPU_UTIL=0.85 PO_JEDAN=1
run() {
  echo "######## BATCH1 $1 gamma=$2 ########"
  METODA=$1 GAMMA=$2 ~/venv-vllm/bin/python scripts/run_eval_vllm.py > ~/n1_$1_$2.txt 2>&1
  grep -E "baseline \(|spekulativno,|vreme |propusnost|ImportError|Error" ~/n1_$1_$2.txt | head -4
  sleep 5
}
run draft_model 0
for G in 3 5 8; do run ngram $G; done
run suffix 5
echo GOTOVO_NGRAM_B1
