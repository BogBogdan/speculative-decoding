#!/bin/bash
# Podesavanje sveze GPU instance za merenja spekulativnog dekodiranja.
#
#   bash scripts/setup_gpu.sh
#
# Softverski stek proveren na: AWS g5.xlarge (A10G 22.5 GB), Ubuntu, NVIDIA
# drajver 595.x, Python 3.14, torch 2.13.0+cu130, transformers 5.15.0, vLLM 0.27.1.
# PAZNJA: sam A10G vise NIJE dovoljan - target je Qwen2.5-14B, kome same tezine
# u bf16 uzimaju ~28 GB, pa merenja idu na karticu od 40 GB navise.
#
# ZAHTEVI PRE POKRETANJA
#   - GPU sa najmanje 40 GB (za Qwen2.5-14B u bf16)
#   - najmanje 110 GB slobodno na disku, vidi racunicu dole
#   - NVIDIA drajveri instalirani (nvidia-smi radi)
#
# ZAUZECE DISKA (venv-ovi izmereni na 7B postavci; 14B stavke su procena)
#   venv (torch)            5.1 GB
#   venv-vllm               8.1 GB
#   HF kes (0.5B + 14B)      ~29 GB
#   draft-151665            953 MB
#   target-151665            ~28 GB
#   -------------------------------
#   ukupno                  ~71 GB   + sistem ~15 GB  =  ~86 GB
#
# Ako je disk manji, prosiri ga u AWS konzoli (Volumes -> Modify Volume) pa:
#   sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1

set -e
REPO="$HOME/speculative-decoding"

echo "### 1. sistemski paketi ###"
sudo apt-get update -qq
# nvidia-cuda-toolkit daje nvcc, bez kojeg vLLM ne moze da pokrene svoj motor
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-venv python3-pip git nvidia-cuda-toolkit
echo "nvcc: $(which nvcc)"

echo "### 2. venv za torch (run_eval.py, run_eval_static.py) ###"
python3 -m venv "$HOME/venv"
"$HOME/venv/bin/pip" install -q --upgrade pip
"$HOME/venv/bin/pip" install -q torch transformers datasets accelerate
"$HOME/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "### 3. venv za vLLM (run_eval_vllm.py) ###"
# ZASEBAN venv: vLLM povlaci svoj CUDA build torcha i pregazio bi gornji.
# ninja i cmake su obavezni - vLLM ih poziva kao spoljne programe pri JIT kompilaciji.
python3 -m venv "$HOME/venv-vllm"
"$HOME/venv-vllm/bin/pip" install -q --upgrade pip
"$HOME/venv-vllm/bin/pip" install -q vllm ninja cmake
"$HOME/venv-vllm/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

echo "### 4. repo i prefiksi ###"
[ -d "$REPO" ] || git clone -q https://github.com/BogBogdan/speculative-decoding.git "$REPO"
mkdir -p "$REPO/data"
"$HOME/venv/bin/python" "$REPO/train/get_dataset.py"

echo "### 5. modeli sa izjednacenim recnikom ###"
# Qwen2.5-0.5B ima vocab_size 151936, a 14B 152064, dok tokenizer ima 151665.
# I HF i vLLM ODBIJAJU par sa razlicitim vocab_size. Oba se smanjuju na 151665 -
# smanjivanje je cisto odsecanje, ne pravi nove redove.
"$HOME/venv/bin/python" "$REPO/scripts/pripremi_modele.py"

cat <<'KRAJ'

########################################################################
GOTOVO. Kako se sta pokrece:

  cd ~/speculative-decoding

  # 1) tvoja implementacija, DynamicCache (najbolja od HF varijanti)
  GEMMA=1 MAX_NOVIH=128 BROJ_PREFIKSA=3 MERI_BASELINE=1 \
    ~/venv/bin/python scripts/run_eval.py

  # 2) tvoja implementacija, StaticCache (COMPILE=1 puca na transformers 5.15)
  GEMMA=1 COMPILE=0 ~/venv/bin/python scripts/run_eval_static.py

  # 3) vLLM - jedina varijanta koja prelazi 1x
  #    PATH mora da sadrzi venv-vllm/bin zbog ninja, CUDA_HOME zbog nvcc
  export PATH="$HOME/venv-vllm/bin:$PATH" CUDA_HOME=/usr
  export DRAFT_ID=$HOME/draft-151665 TARGET_ID=$HOME/target-151665
  PO_JEDAN=1 GAMMA=0 ~/venv-vllm/bin/python scripts/run_eval_vllm.py   # baseline
  PO_JEDAN=1 GAMMA=1 ~/venv-vllm/bin/python scripts/run_eval_vllm.py   # spekulativno

  # ili sve odjednom:
  PO_JEDAN=1 bash scripts/run_vllm.sh

PO_JEDAN=1 znaci batch 1 (kasnjenje). Bez toga vLLM grupise sve prefikse i
meri propusnost, sto nije uporedivo sa run_eval.py.
########################################################################
KRAJ
