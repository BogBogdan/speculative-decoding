"""Izjednaci vocab_size draft i target modela, snimi lokalno.

Qwen2.5 porodica ima razlicite velicine matrice ugnjezdenja:
    0.5B / 1.5B / 3B   vocab_size = 151936
    7B / 14B / 32B     vocab_size = 152064
    tokenizer                     = 151665 stvarnih tokena

Visak su neistrenirani redovi koji ne dekodiraju ni u sta (masa verovatnoce
na njima je ~1e-9). I transformers i vLLM ODBIJAJU spekulativno dekodiranje
dok se vocab_size ne izjednaci.

Oba modela se smanjuju na len(tokenizer). Smanjivanje je cisto odsecanje -
za razliku od prosirivanja, koje bi novim redovima dalo nasumicne vrednosti
i obaralo alfa.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_ID = os.environ.get("DRAFT_ID", "Qwen/Qwen2.5-0.5B")
TARGET_ID = os.environ.get("TARGET_ID", "Qwen/Qwen2.5-14B")
IZLAZ_DRAFT = os.environ.get("IZLAZ_DRAFT", os.path.expanduser("~/draft-151665"))
IZLAZ_TARGET = os.environ.get("IZLAZ_TARGET", os.path.expanduser("~/target-151665"))

tok = AutoTokenizer.from_pretrained(DRAFT_ID)
V = len(tok)
print(f"ciljni recnik (len tokenizer) = {V}")

for ime, put in [(DRAFT_ID, IZLAZ_DRAFT), (TARGET_ID, IZLAZ_TARGET)]:
    if os.path.isdir(put):
        print(f"{put} vec postoji, preskacem")
        continue
    m = AutoModelForCausalLM.from_pretrained(ime, dtype=torch.bfloat16)
    pre = m.config.vocab_size
    if pre < V:
        raise SystemExit(f"{ime} ima vocab_size {pre} < {V}; prosirivanje nije bezbedno")
    m.resize_token_embeddings(V)
    m.save_pretrained(put)
    tok.save_pretrained(put)
    print(f"{ime}: {pre} -> {m.config.vocab_size}  ->  {put}")
    del m

print("gotovo")
