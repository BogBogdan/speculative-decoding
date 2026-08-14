"""Destilacioni podaci ciljano za prefikse na kojima ce se meriti alfa.

Iz svakog prefiksa se generise VISE nastavaka. Posto je uzorkovanje stohasticno,
svaki prolaz daje drugu granu - pa student uci RASPODELU ucitelja u toj tacki,
a ne jedan put kroz nju. To je ono sto alfa i meri.

Format izlaza je isti kao kod distilation.ipynb, pa DistillationDataset i
train.py rade bez izmena.

VAZNO: GEN_LEN ovde mora da se poklopi sa GEN_LEN u train.py.

    python train/gen_demo_dataset.py
    GEN_LEN=128 PO_PREFIKSU=50 python train/gen_demo_dataset.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("USER", "aleksa")
os.environ.setdefault("HF_HOME", "/data")

import torch

sys.path.append("/home/mls07/speculative-decoding/src")
from model import load_teacher, load_tokenizer

PREFIKSI = os.environ.get("PREFIKSI", str(Path(__file__).resolve().parent / "prefiksi.pt"))
IZLAZ_DIR = os.environ.get("IZLAZ_DIR", "/home/mls07/data-demo")
GEN_LEN = int(os.environ.get("GEN_LEN", 128))        # mora = GEN_LEN u train.py
PO_PREFIKSU = int(os.environ.get("PO_PREFIKSU", 200))  # koliko nastavaka po prefiksu;
# vise nastavaka = sire pokriveno stablo grana, pa alfa manje opada kroz generisanje
BATCH = int(os.environ.get("BATCH", 8))              # koliko odjednom na kartici
TOP_K = int(os.environ.get("TOP_K", 50))

os.makedirs(IZLAZ_DIR, exist_ok=True)
tokenizer = load_tokenizer()
teacher = load_teacher()

prefiksi = torch.load(PREFIKSI, weights_only=True)
# isto ime kao u eval skriptama, da se generisanje i merenje rade nad istim brojem
BROJ_PREFIKSA = int(os.environ.get("BROJ_PREFIKSA", 3))
prefiksi = prefiksi[:BROJ_PREFIKSA]

print(f"{len(prefiksi)} prefiksa, {PO_PREFIKSU} nastavaka po prefiksu, "
      f"{GEN_LEN} tokena po nastavku")
print(f"ukupno {len(prefiksi) * PO_PREFIKSU} sekvenci -> {IZLAZ_DIR}\n")


@torch.no_grad()
def generisi(prefiks_1d, koliko):
    """koliko nastavaka iz istog prefiksa. Isti prefiks ponovljen u batchu daje
    razlicite nastavke jer je do_sample=True."""
    ulaz = prefiks_1d.unsqueeze(0).repeat(koliko, 1).to(teacher.device)
    izlaz = teacher.generate(
        ulaz,
        attention_mask=torch.ones_like(ulaz),
        max_new_tokens=GEN_LEN,
        min_new_tokens=GEN_LEN,       # bez ranog EOS-a -> nema dopune koja kvari ciljeve
        do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0,
        output_logits=True, return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    vals, idx = [], []
    for logits in izlaz.logits:                       # tuple duzine GEN_LEN
        v, i = torch.topk(logits, TOP_K, dim=-1)
        vals.append(v.to(torch.float16).cpu())
        idx.append(i.to(torch.int32).cpu())
    return {
        "prefix_ids": ulaz.cpu(),
        "generated_ids": izlaz.sequences.cpu(),
        "topk_logits": torch.stack(vals, dim=1),
        "topk_indices": torch.stack(idx, dim=1),
    }


n = 0
for p_idx, p in enumerate(prefiksi):
    p = p.squeeze(0) if p.dim() == 2 else p
    preostalo = PO_PREFIKSU
    while preostalo > 0:
        koliko = min(BATCH, preostalo)
        put = os.path.join(IZLAZ_DIR, f"batch_{n:06d}.pt")
        if not os.path.exists(put):
            torch.save(generisi(p, koliko), put)
        n += koliko
        preostalo -= koliko
    print(f"prefiks {p_idx + 1}/{len(prefiksi)} gotov  ({n} sekvenci)")

print(f"\ngotovo: {n} sekvenci u {IZLAZ_DIR}")
print(f"u train.py stavi DATA_DIR = \"{IZLAZ_DIR}\" i GEN_LEN = {GEN_LEN}")
