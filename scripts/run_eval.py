import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, DynamicCache
from src.speculative import speculative_sampling
from src.metrics import Merenja, izmeri_c, odstupanje_raspodele, perplexity, autoregresivno

draft  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").eval()
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").eval() #bice 14B ovo je fora
gemma = 5  # parametar sto predstavlja broj tokena koji draft generise 

MAX_NOVIH = 128        # koliko tokena generisati po prefiksu
BROJ_PREFIKSA = 20     # koliko prefiksa proci
MERI_BASELINE = True   # obicno dekodiranje radi izmerenog ubrzanja (duplira vreme)


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
            q = torch.softmax(izlaz.logits[0, -1].float(), dim=-1)
            x = torch.multinomial(q, 1).item()
            q_lista.append(q)
            x_lista.append(x)
            prosireni = torch.cat([prosireni, torch.tensor([[x]], dtype=ids.dtype, device=ids.device)], dim=1)

        novi = prosireni[:, target_cache.get_seq_length():]
        izlaz = target(novi, past_key_values=target_cache, use_cache=True)
        target_cache = izlaz.past_key_values
        p_logits = izlaz.logits[0, -(gemma + 1):].float()
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

    # Theorem 1: svaki par (p_i, q_i) mora da da tacno p_i kao izlaznu raspodelu
    odstupanje = max(odstupanje_raspodele(p_lista[j], q_lista[j]) for j in range(gemma))

    return ids, n, odstupanje, draft_cache, target_cache


if __name__ == "__main__":
    prefiksi = torch.load(ROOT / "data" / "prefiksi.pt", weights_only=True)
    m = Merenja(gemma)

    prvi = prefiksi[0]
    if prvi.dim() != 2:
        prvi = prvi.unsqueeze(0)
    c, t_draft, t_target = izmeri_c(draft, target, prvi)

    t_baseline = 0.0
    koliko = min(BROJ_PREFIKSA, len(prefiksi))

    for idx in range(koliko):
        ids = prefiksi[idx]
        if ids.dim() != 2:
            ids = ids.unsqueeze(0)
        pocetna_duzina = ids.shape[1]
        draft_cache = DynamicCache()
        target_cache = DynamicCache()

        t0 = time.perf_counter()
        while ids.shape[1] - pocetna_duzina < MAX_NOVIH:
            ids, n, odstupanje, draft_cache, target_cache = speculativni_korak(
                ids, draft_cache, target_cache
            )
            m.dodaj(n, odstupanje)
        m.vreme += time.perf_counter() - t0

        if MERI_BASELINE:
            osnova = prefiksi[idx]
            if osnova.dim() != 2:
                osnova = osnova.unsqueeze(0)
            t0 = time.perf_counter()
            autoregresivno(target, osnova, MAX_NOVIH)
            t_baseline += time.perf_counter() - t0

        print(f"prefiks {idx + 1}/{koliko} gotov")

    r = m.rezultat(c=c, t_baseline=t_baseline if MERI_BASELINE else None)

    print(f"\ngamma = {gemma}, prefiksa = {koliko}, tokena po prefiksu = {MAX_NOVIH}")
    print(f"  alfa (acceptance rate)        : {r['alfa']:.4f}")
    print(f"  c (cost efficiency)           : {r['c']:.4f}"
          f"   [draft {t_draft * 1000:.1f} ms, target {t_target * 1000:.1f} ms]")
    print(f"  tokena / poziv target modela  : {r['tokena_po_pozivu_target']:.4f}"
          f"   (teorijski {r['teorijski_tokena_po_pozivu']:.4f})")
    print(f"  teorijsko ubrzanje            : {r['teorijsko_ubrzanje']:.4f}x")
    if MERI_BASELINE:
        print(f"  izmereno ubrzanje             : {r['izmereno_ubrzanje']:.4f}x"
              f"   [spec {m.vreme:.1f} s, baseline {t_baseline:.1f} s]")
    print(f"  max odstupanje raspodele      : {r['max_odstupanje_raspodele']:.2e}"
          f"   (Theorem 1, ocekuje se ~1e-7)")
    print(f"  raspodela n                   : {m.histogram_n}")

    uzorak = [prefiksi[i] for i in range(min(50, len(prefiksi)))]
    print(f"  perplexity draft modela       : {perplexity(draft, uzorak):.4f}")
    print(f"  perplexity target modela      : {perplexity(target, uzorak):.4f}   (donja granica)")
