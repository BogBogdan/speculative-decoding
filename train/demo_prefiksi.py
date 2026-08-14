"""Napravi prefiksi.pt od prefiksa koje je koristio ucitelj pri generisanju.

Za DEMO: student je destilovan bas na uciteljevim nastavcima iz ovih tacaka,
pa je alfa na njima najveca moguca. Na novim prefiksima efekta nece biti -
ovo meri koliko je student zapamtio, ne koliko je naucio.

Za rad koristi scripts/get_dataset.py, koji uzima wikitext TEST split.

    python train/demo_prefiksi.py /home/mls07/data 50 100
                                  ^data_dir      ^koliko ^duzina
"""
import glob
import os
import sys

import torch

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/mls07/data"
KOLIKO = int(sys.argv[2]) if len(sys.argv) > 2 else 50
DUZINA = int(sys.argv[3]) if len(sys.argv) > 3 else 100
IZLAZ = os.environ.get("IZLAZ", "data/prefiksi-demo.pt")

putanje = sorted(glob.glob(os.path.join(DATA_DIR, "batch_*.pt")))
if not putanje:
    raise SystemExit(f"nema batch_*.pt u {DATA_DIR}")

prefiksi = []
for p in putanje:
    b = torch.load(p, map_location="cpu", weights_only=True)
    if "prefix_ids" not in b:
        raise SystemExit(f"{p} nema prefix_ids - regenerisi podatke novijim notebookom")
    for red in b["prefix_ids"]:
        if red.shape[-1] < DUZINA:
            continue
        prefiksi.append(red[:DUZINA].unsqueeze(0))    # [1, DUZINA], kao get_dataset.py
        if len(prefiksi) >= KOLIKO:
            break
    if len(prefiksi) >= KOLIKO:
        break

os.makedirs(os.path.dirname(IZLAZ) or ".", exist_ok=True)
torch.save(prefiksi, IZLAZ)
print(f"{len(prefiksi)} prefiksa x {DUZINA} tokena -> {IZLAZ}")
print("koristi ga sa: cp", IZLAZ, "data/prefiksi.pt")
