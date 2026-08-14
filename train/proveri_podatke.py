"""Dijagnostika destilacionih podataka - zasto destilovani student kolabira.

Pokrenuti na masini gde su batch_*.pt fajlovi:
    python train/proveri_podatke.py /home/mls07/data
"""
import glob
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/mls07/data"
STUDENT = os.environ.get("STUDENT", "Qwen/Qwen2.5-0.5B")
TEACHER = os.environ.get("TEACHER", "Qwen/Qwen2.5-14B")

V_STUDENT = AutoConfig.from_pretrained(STUDENT).vocab_size
V_TEACHER = AutoConfig.from_pretrained(TEACHER).vocab_size
tok = AutoTokenizer.from_pretrained(STUDENT)
EOS = tok.eos_token_id

print(f"student {STUDENT}: vocab_size = {V_STUDENT}")
print(f"teacher {TEACHER}: vocab_size = {V_TEACHER}")
print(f"eos_token_id = {EOS}, len(tokenizer) = {len(tok)}\n")

putanje = sorted(glob.glob(os.path.join(DATA_DIR, "batch_*.pt")))
if not putanje:
    raise SystemExit(f"nema batch_*.pt u {DATA_DIR}")
print(f"nadjeno {len(putanje)} fajlova, gledam prvih {min(20, len(putanje))}\n")

van_opsega = 0
ukupno_idx = 0
eos_poz = 0
ukupno_poz = 0
top1_ver = []
duzine = set()
najveci_idx = 0

for p in putanje[:20]:
    b = torch.load(p, map_location="cpu")
    gi, tl, ti = b["generated_ids"], b["topk_logits"], b["topk_indices"]
    duzine.add((tuple(gi.shape[1:]), tuple(tl.shape[1:]), tuple(ti.shape[1:])))

    # 1. indeksi van studentovog recnika -> gather cita van granica
    idx = ti.long()
    najveci_idx = max(najveci_idx, int(idx.max()))
    van_opsega += int((idx >= V_STUDENT).sum())
    ukupno_idx += idx.numel()

    # 2. koliko ciljnih tokena je EOS (dopuna posle ranog zavrsetka)
    gen = gi[:, -tl.shape[1]:]          # samo generisani deo
    eos_poz += int((gen == EOS).sum())
    ukupno_poz += gen.numel()

    # 3. koliko je uciteljeva raspodela ostra
    p_uc = torch.softmax(tl.float(), dim=-1)
    top1_ver.append(p_uc[..., 0].flatten())

top1 = torch.cat(top1_ver)

print("=" * 60)
print("1. INDEKSI VAN STUDENTOVOG RECNIKA")
print(f"   najveci indeks u podacima : {najveci_idx}  (granica {V_STUDENT})")
print(f"   van opsega                : {van_opsega} / {ukupno_idx}"
      f"  ({100 * van_opsega / ukupno_idx:.6f}%)")
if van_opsega:
    print("   >>> OVO JE BAG. gather cita van granica, gradijenti su smece.")
    print("   >>> Popravka: odseci na min(V_STUDENT, len(tokenizer)) pri generisanju,")
    print("       ili filtriraj indekse >= V_STUDENT pre kd_loss.")
else:
    print("   OK - nijedan indeks ne prelazi studentov recnik")

print()
print("2. EOS U CILJEVIMA (dopuna posle ranog zavrsetka generate-a)")
print(f"   EOS pozicija : {eos_poz} / {ukupno_poz}  ({100 * eos_poz / ukupno_poz:.2f}%)")
if eos_poz / ukupno_poz > 0.05:
    print("   >>> SUMNJIVO. Student uci da izbacuje EOS na tim pozicijama.")
    print("   >>> Popravka: min_new_tokens=GEN_LEN pri generate, ili maska preko EOS repa.")
else:
    print("   OK - zanemarljivo")

print()
print("3. OSTRINA UCITELJEVE RASPODELE (top-1 posle renormalizacije na top-50)")
print(f"   prosek : {top1.mean():.4f}")
print(f"   udeo > 0.99 : {100 * (top1 > 0.99).float().mean():.2f}%")
if (top1 > 0.99).float().mean() > 0.5:
    print("   >>> Raspodela je skoro jednotackasta; KD se svodi na obicno ucenje 1 tokena.")

print()
print("4. OBLICI TENZORA")
for d in duzine:
    print(f"   generated_ids[1:]={d[0]}  topk_logits[1:]={d[1]}  topk_indices[1:]={d[2]}")
if len(duzine) > 1:
    print("   >>> Oblici se razlikuju izmedju fajlova - poravnanje nije garantovano.")
print("=" * 60)
