"""Prefiksi za destilaciju i merenje alfa, iz wikitext-103 test splita.

Sve sto pipeline treba stoji pored: izlaz se snima u train/prefiksi.pt, pa
gen_demo_dataset.py i run_eval.py citaju odatle.

Putanje su izvedene iz __file__, pa je svejedno odakle se skripta pokrece.

    python train/get_dataset.py
    KOLIKO=3 DUZINA=128 python train/get_dataset.py
"""
import os
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

OVDE = Path(__file__).resolve().parent
IZLAZ = Path(os.environ.get("PREFIKSI", OVDE / "prefiksi.pt"))
DRAFT_ID = os.environ.get("DRAFT_ID", "Qwen/Qwen2.5-0.5B")
KOLIKO = int(os.environ.get("KOLIKO", 100))    # koliko pasusa uzeti
DUZINA = int(os.environ.get("DUZINA", 50))     # tokena po prefiksu
SEED = int(os.environ.get("SEED", 42))

tok = AutoTokenizer.from_pretrained(DRAFT_ID)


def get_dataset():
    return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")


def process_row(row_text):
    """Ponistava tokenizacione artefakte wikitexta. Naslovi i kratki pasusi ispadaju."""
    if row_text.startswith(" = ") or len(row_text) < 300:
        return None

    t = row_text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    t = re.sub(r" ([,.;:!?%])", r"\1", t)
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r"\$ ", "$", t)
    return t


def get_prefix():
    return [t for row in get_dataset() if (t := process_row(row["text"])) is not None]


def uzorak_pasusa(pasusi, koliko, seed=SEED):
    """Nasumican izbor KOLIKO pasusa. Fiksan seed - isti prefiksi pri svakom pokretanju,
    inace merenje pre i posle destilacije ne bi bilo nad istim tekstom."""
    if len(pasusi) <= koliko:
        return pasusi
    return random.Random(seed).sample(pasusi, koliko)


def tokenizuj_prefikse(pasusi, duzina=DUZINA):
    """Svaki pasus se odseca na tacno `duzina` tokena. Kraci se izbacuju, da svi
    prefiksi budu iste duzine - inace bi merenja vremena bila neuporediva."""
    out = []
    for t in pasusi:
        ids = tok(t, return_tensors="pt").input_ids[:, :duzina]
        if ids.shape[-1] == duzina:
            out.append(ids)
    return out


if __name__ == "__main__":
    svi = get_prefix()
    uzorak = uzorak_pasusa(svi, KOLIKO)
    ids = tokenizuj_prefikse(uzorak)

    IZLAZ.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, IZLAZ)

    print(f"pasusa u test splitu posle filtriranja : {len(svi)}")
    print(f"uzorkovano                             : {len(uzorak)}")
    print(f"tokenizovano na {DUZINA} tokena          : {len(ids)}")
    print(f"snimljeno                              : {IZLAZ}")
