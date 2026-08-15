
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from src.speculative import speculative_sampling
from src.metrics import Merenja, autoregresivno

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
DRAFT_ID = os.environ.get("DRAFT_ID", "Qwen/Qwen2.5-0.5B")
TARGET_ID = os.environ.get("TARGET_ID", "Qwen/Qwen2.5-14B")

gemma = int(os.environ.get("GEMMA", 5))
MAX_NOVIH = int(os.environ.get("MAX_NOVIH", 128))
PREFIKSI = os.environ.get("PREFIKSI", str(ROOT / "train" / "prefiksi.pt"))
BROJ_PREFIKSA = int(os.environ.get("BROJ_PREFIKSA", 3))
MERI_BASELINE = os.environ.get("MERI_BASELINE", "1") == "1"
COMPILE = os.environ.get("COMPILE", "1") == "1"

tok = AutoTokenizer.from_pretrained(DRAFT_ID)
V = len(tok)   # 151665 stvarnih tokena; sve preko toga je dopuna u matrici ugnjezdenja
EOS_ID = tok.eos_token_id

draft = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=DTYPE).to(DEVICE).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=DTYPE).to(DEVICE).eval()

if COMPILE:
    # reduce-overhead ukljucuje CUDA grafove; smisleno je samo uz fiksne oblike
    draft = torch.compile(draft, mode="reduce-overhead")
    target = torch.compile(target, mode="reduce-overhead")


def nova_kes(model, max_len):
    return StaticCache(config=model.config, max_cache_len=max_len,
                       max_batch_size=1, device=DEVICE, dtype=DTYPE)


def vrati_na(kes, duzina):
    """Rollback: StaticCache nema crop, ali pise na cumulative_length.
    Zaostali unosi iznad te duzine se prepisu sledecim tokenima i ne gledaju se."""
    for sloj in kes.layers:
        cl = sloj.cumulative_length
        if torch.is_tensor(cl):
            cl.fill_(duzina)
        else:
            sloj.cumulative_length = duzina


_MARK = getattr(torch.compiler, "cudagraph_mark_step_begin", None)


def pozovi(model, kes, prosireni, vidjeno):
    """Posalji modelu samo ono sto jos nije video. Vraca (logits, nova_duzina)."""
    novi = prosireni[:, vidjeno:]
    pozicije = torch.arange(vidjeno, prosireni.shape[1], device=prosireni.device)
    # CUDA grafovi recikliraju izlazni bafer; bez ovoga sledeci poziv prepise
    # q raspodele koje cuvamo do verifikacije
    if COMPILE and _MARK is not None:
        _MARK()
    izlaz = model(novi, past_key_values=kes, use_cache=True, cache_position=pozicije)
    return izlaz.logits, prosireni.shape[1]


def speculativni_korak(ids, draft_kes, target_kes, d_len, t_len):
    """Jedna iteracija Algorithm 1. Vraca (ids, n, d_len, t_len)."""
    L = ids.shape[1]
    prosireni = ids
    q_lista, x_lista = [], []

    with torch.no_grad():
        # draft: gamma nagadjanja, posle prve iteracije uvek po 1 token
        for _ in range(gemma):
            logits, d_len = pozovi(draft, draft_kes, prosireni, d_len)
            # clone jer se ovo cuva kroz sledece pozive modela
            q = torch.softmax(logits[0, -1, :V].float(), dim=-1).clone()
            x = torch.multinomial(q, 1).item()
            q_lista.append(q)
            x_lista.append(x)
            prosireni = torch.cat(
                [prosireni, torch.tensor([[x]], dtype=ids.dtype, device=ids.device)], dim=1)

        # target: jedan poziv, posle prve iteracije uvek tacno gemmjea+1 pozicija
        logits, t_len = pozovi(target, target_kes, prosireni, t_len)
        p_lista = torch.softmax(logits[0, -(gemma + 1):, :V].float(), dim=-1).clone()

    # verifikacija
    n = gemma
    for i in range(gemma):
        if not speculative_sampling(p_lista[i], q_lista[i], x_lista[i]):
            n = i
            break

    prihvaceni = x_lista[:n]

    # bonus token ako je sve proslo, inace korekcija odbijenog
    if n == gemma:
        sledeci = torch.multinomial(p_lista[gemma], 1).item()
    else:
        backup = (p_lista[n] - q_lista[n]).clamp(min=0)
        sledeci = torch.multinomial(backup / backup.sum(), 1).item()

    ids = torch.cat(
        [ids, torch.tensor([prihvaceni + [sledeci]], dtype=ids.dtype, device=ids.device)], dim=1)

    # rollback: vazi samo do L+n. draft nikad nije video svoje poslednje nagadjanje,
    # pa mu je duzina najvise L+gemma-1.
    d_len = min(d_len, L + n)
    t_len = min(t_len, L + n)
    vrati_na(draft_kes, d_len)
    vrati_na(target_kes, t_len)

    return ids, n, d_len, t_len


if __name__ == "__main__":
    prefiksi = torch.load(PREFIKSI, weights_only=True)
    koliko = min(BROJ_PREFIKSA, len(prefiksi))
    m = Merenja(gemma)
    t_baseline = 0.0
    tokena_baseline = 0

    prvi = prefiksi[0]
    if prvi.dim() == 1:
        prvi = prvi.unsqueeze(0)
    MAX_KES = prvi.shape[1] + MAX_NOVIH + gemma + 8

    for idx in range(koliko):
        ids = prefiksi[idx].to(DEVICE)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        pocetna_duzina = ids.shape[1]

        draft_kes = nova_kes(draft, MAX_KES)
        target_kes = nova_kes(target, MAX_KES)
        d_len = t_len = 0

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        while ids.shape[1] - pocetna_duzina < MAX_NOVIH:
            ids, n, d_len, t_len = speculativni_korak(ids, draft_kes, target_kes, d_len, t_len)
            m.dodaj(n)
            if EOS_ID is not None and (ids[0, pocetna_duzina:] == EOS_ID).any():
                break
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        m.vreme += time.perf_counter() - t0

        if MERI_BASELINE:
            osnova = prefiksi[idx].to(DEVICE)
            if osnova.dim() == 1:
                osnova = osnova.unsqueeze(0)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            izlaz_base = autoregresivno(target, osnova, MAX_NOVIH, V=V, eos_id=EOS_ID)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t_baseline += time.perf_counter() - t0
            tokena_baseline += izlaz_base.shape[1] - osnova.shape[1]

        print(f"prefiks {idx + 1}/{koliko} gotov")

    # svedi baseline na isti broj tokena pre poredjenja
    t_base_norm = t_baseline / tokena_baseline * m.tokena if tokena_baseline else None
    r = m.rezultat(t_baseline=t_base_norm)
    print(f"\nStaticCache | compile={COMPILE} | gamma={gemma}, prefiksa={koliko}, "
          f"tokena po prefiksu={MAX_NOVIH}")
    print(f"  alfa                          : {r['alfa']:.4f}")
    print(f"  tokena / poziv target modela  : {r['tokena_po_pozivu_target']:.4f}"
          f"   (teorijski {r['teorijski_tokena_po_pozivu']:.4f})")
    if MERI_BASELINE:
        print(f"  izmereno ubrzanje             : {r['izmereno_ubrzanje']:.4f}x"
              f"   [spec {m.vreme:.1f} s, baseline {t_baseline:.1f} s]")
    print(f"  raspodela n                   : {m.histogram_n}")
