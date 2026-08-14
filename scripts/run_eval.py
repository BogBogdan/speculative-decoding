import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from src.speculative import speculative_sampling
from src.metrics import Merenja, izmeri_c, perplexity, autoregresivno

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
DRAFT_ID = os.environ.get("DRAFT_ID", "Qwen/Qwen2.5-0.5B")
TARGET_ID = os.environ.get("TARGET_ID", "Qwen/Qwen2.5-7B")

tok = AutoTokenizer.from_pretrained(DRAFT_ID)
V = len(tok)          # 151665 stvarnih tokena; 0.5B ima 151936 a 7B/14B 152064 mesta,
EOS_ID = tok.eos_token_id   # visak su neistrenirani redovi koji ne dekodiraju ni u sta

draft  = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=DTYPE).to(DEVICE).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=DTYPE).to(DEVICE).eval()
gemma = int(os.environ.get("GEMMA", 5))  # broj tokena koji draft generise po iteraciji

MAX_NOVIH = int(os.environ.get("MAX_NOVIH", 128))          # tokena po prefiksu
PREFIKSI = os.environ.get("PREFIKSI", str(ROOT / "train" / "prefiksi.pt"))
BROJ_PREFIKSA = int(os.environ.get("BROJ_PREFIKSA", 3))   # koliko prefiksa proci
MERI_BASELINE = os.environ.get("MERI_BASELINE", "1") == "1"


def speculativni_korak(ids, draft_cache, target_cache):

    #logiti ucitavanja i lista tokena
    pocetna_duzina = ids.shape[1]
    prosireni = ids
    q_lista = []
    x_lista = []
    with torch.no_grad():
        for i in range(gemma):
            # posalji samo ono sto draft jos nije video (prvi put ceo prefiks, dalje 1 token)
            novi = prosireni[:, draft_cache.get_seq_length():]
            izlaz = draft(novi, past_key_values=draft_cache, use_cache=True)
            draft_cache = izlaz.past_key_values
            q = torch.softmax(izlaz.logits[0, -1, :V].float(), dim=-1)
            x = torch.multinomial(q, 1).item()
            q_lista.append(q)
            x_lista.append(x)
            prosireni = torch.cat([prosireni, torch.tensor([[x]], dtype=ids.dtype, device=ids.device)], dim=1)

        novi = prosireni[:, target_cache.get_seq_length():]
        izlaz = target(novi, past_key_values=target_cache, use_cache=True)
        target_cache = izlaz.past_key_values
        # odsecanje na V ide PRE softmaxa, da se raspodela normalizuje preko pravih tokena
        p_logits = izlaz.logits[0, -(gemma + 1):, :V].float()
        p_lista = torch.softmax(p_logits, dim=-1)

    #verifikacija bolje da se ne paralelizuje malo je gama
    n = gemma
    for i in range(gemma):
        p, q, x = p_lista[i], q_lista[i], x_lista[i]
        if not speculative_sampling(p, q, x):
            n = i
            break

    prihvaceni = x_lista[:n]

    #bonus token bukvalno u radu pise da se daje bonus token ako sve prodje dobro idk(Vec je izracunat)
    if n == gemma:
        sledeci = torch.multinomial(p_lista[gemma], 1).item()
    else:
        backup = (p_lista[n] - q_lista[n]).clamp(min=0)
        backup = backup / backup.sum()
        sledeci = torch.multinomial(backup, 1).item()

    ids = torch.cat(
        [ids, torch.tensor([prihvaceni + [sledeci]], dtype=ids.dtype, device=ids.device)], dim=1
    )

    # odbaceni tokeni su vec upisani u oba cache-a,
    # inace sledeca iteracija racuna paznju nad tokenima kojih vise nema
    for cache in (draft_cache, target_cache):
        visak = cache.get_seq_length() - (pocetna_duzina + n)
        if visak > 0:
            cache.crop(-visak)

    return ids, n, draft_cache, target_cache


if __name__ == "__main__":
    prefiksi = torch.load(PREFIKSI, weights_only=True)
    m = Merenja(gemma)

    prvi = prefiksi[0].to(DEVICE)
    if prvi.dim() != 2:
        prvi = prvi.unsqueeze(0)
    c, t_draft, t_target = izmeri_c(draft, target, prvi)

    t_baseline = 0.0
    tokena_baseline = 0
    koliko = min(BROJ_PREFIKSA, len(prefiksi))

    def sinhronizuj():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    for idx in range(koliko):
        ids = prefiksi[idx].to(DEVICE)
        if ids.dim() != 2:
            ids = ids.unsqueeze(0)
        pocetna_duzina = ids.shape[1]
        draft_cache = DynamicCache()
        target_cache = DynamicCache()

        sinhronizuj()
        t0 = time.perf_counter()
        while ids.shape[1] - pocetna_duzina < MAX_NOVIH:
            ids, n, draft_cache, target_cache = speculativni_korak(
                ids, draft_cache, target_cache
            )
            m.dodaj(n)
            if EOS_ID is not None and (ids[0, pocetna_duzina:] == EOS_ID).any():
                break
        sinhronizuj()
        m.vreme += time.perf_counter() - t0

        if MERI_BASELINE:
            osnova = prefiksi[idx].to(DEVICE)
            if osnova.dim() != 2:
                osnova = osnova.unsqueeze(0)
            sinhronizuj()
            t0 = time.perf_counter()
            izlaz_base = autoregresivno(target, osnova, MAX_NOVIH, V=V, eos_id=EOS_ID)
            sinhronizuj()
            t_baseline += time.perf_counter() - t0
            tokena_baseline += izlaz_base.shape[1] - osnova.shape[1]

        print(f"prefiks {idx + 1}/{koliko} gotov")

    # baseline i spekulativno ne daju isti broj tokena (prebacaj, EOS), pa se
    # vreme baseline-a svede na isti broj tokena pre poredjenja
    t_base_norm = None
    if MERI_BASELINE and tokena_baseline:
        t_base_norm = t_baseline / tokena_baseline * m.tokena

    r = m.rezultat(c=c, t_baseline=t_base_norm)

    print(f"\ngamma = {gemma}, prefiksa = {koliko}, tokena po prefiksu = {MAX_NOVIH}")
    print(f"  uredjaj / dtype               : {DEVICE} / {DTYPE}")
    print(f"  draft / target                : {DRAFT_ID} / {TARGET_ID}")
    print(f"  alfa (acceptance rate)        : {r['alfa']:.4f}")
    print(f"  c (cost efficiency)           : {r['c']:.4f}"
          f"   [draft {t_draft * 1000:.1f} ms, target {t_target * 1000:.1f} ms]")
    print(f"  tokena / poziv target modela  : {r['tokena_po_pozivu_target']:.4f}"
          f"   (teorijski {r['teorijski_tokena_po_pozivu']:.4f})")
    print(f"  teorijsko ubrzanje            : {r['teorijsko_ubrzanje']:.4f}x")
    if MERI_BASELINE:
        print(f"  izmereno ubrzanje             : {r['izmereno_ubrzanje']:.4f}x"
              f"   [spec {m.vreme:.1f} s / {m.tokena} tok, "
              f"baseline {t_baseline:.1f} s / {tokena_baseline} tok]")
    print(f"  raspodela n                   : {m.histogram_n}")

    uzorak = [prefiksi[i] for i in range(min(50, len(prefiksi)))]
    print(f"  perplexity draft modela       : {perplexity(draft, uzorak):.4f}")
    print(f"  perplexity target modela      : {perplexity(target, uzorak):.4f}   (donja granica)")
