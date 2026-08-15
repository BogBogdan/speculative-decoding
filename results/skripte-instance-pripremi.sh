#!/bin/bash
rm -rf ~/qwen05b-pad
~/venv/bin/python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
V = len(tok)
print("ciljni recnik (len tokenizer):", V)
for ime, put in [("Qwen/Qwen2.5-0.5B", "/home/ubuntu/draft-151665"),
                 ("Qwen/Qwen2.5-14B",  "/home/ubuntu/target-151665")]:
    m = AutoModelForCausalLM.from_pretrained(ime, dtype=torch.bfloat16)
    pre = m.config.vocab_size
    m.resize_token_embeddings(V)          # naniže = cisto odsecanje, bez novih redova
    m.save_pretrained(put)
    tok.save_pretrained(put)
    print(f"{ime}: {pre} -> {m.config.vocab_size}  ->  {put}")
    del m
PY
du -sh ~/draft-151665 ~/target-151665
df -h / | tail -1
echo GOTOVO_PRIPREMA
