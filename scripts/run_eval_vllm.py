"""Merenje ubrzanja spekulativnog dekodiranja unutar vLLM-a.

Poredi se vLLM bez spekulacije naspram vLLM sa spekulacijom, isti stack,
isti modeli, isti prefiksi - pa je razlika iskljucivo algoritam.

Jedna konfiguracija po pokretanju, jer vLLM pri pravljenju LLM objekta zauzme
KV cache; vise instanci u istom procesu obara memoriju. Petlja je u run_vllm.sh.

  GAMMA=0            -> baseline, samo target model
  GAMMA=5            -> spekulativno sa 5 nagadjanja

Primer:
  GAMMA=0 python scripts/run_eval_vllm.py
  GAMMA=5 python scripts/run_eval_vllm.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from vllm import LLM, SamplingParams

DRAFT_ID = os.environ.get("DRAFT_ID", "Qwen/Qwen2.5-0.5B")
TARGET_ID = os.environ.get("TARGET_ID", "Qwen/Qwen2.5-7B")
GAMMA = int(os.environ.get("GAMMA", 0))          # 0 = bez spekulacije
MAX_NOVIH = int(os.environ.get("MAX_NOVIH", 128))
BROJ_PREFIKSA = int(os.environ.get("BROJ_PREFIKSA", 20))
GPU_UTIL = float(os.environ.get("GPU_UTIL", 0.85))
SEED = int(os.environ.get("SEED", 0))


def ucitaj_prefikse():
    p = torch.load(ROOT / "data" / "prefiksi.pt", weights_only=True)
    out = []
    for t in p[:BROJ_PREFIKSA]:
        if t.dim() == 2:
            t = t[0]
        out.append(t.tolist())
    return out


def napravi_llm():
    kw = dict(model=TARGET_ID, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL,
              max_model_len=1024, seed=SEED, enforce_eager=False)
    if GAMMA > 0:
        # noviji vLLM: speculative_config; stariji: speculative_model / num_speculative_tokens
        try:
            return LLM(speculative_config={"model": DRAFT_ID,
                                           "num_speculative_tokens": GAMMA}, **kw)
        except TypeError:
            return LLM(speculative_model=DRAFT_ID, num_speculative_tokens=GAMMA, **kw)
    return LLM(**kw)


def prijavi_alfa(llm):
    """Stopa prihvatanja iz vLLM metrika. Naziv se menjao kroz verzije,
    pa je najbolji trud - treba da bude oko 0.66 za par 0.5B/7B."""
    try:
        for izvor in (getattr(llm, "llm_engine", None), llm):
            for ime in ("get_metrics", "_get_stats", "get_stats"):
                f = getattr(izvor, ime, None)
                if f is None:
                    continue
                mets = f()
                tekst = str(mets)
                if "accept" in tekst.lower():
                    return tekst[:400]
    except Exception as e:
        return f"(nedostupno: {type(e).__name__})"
    return "(nije nadjeno u metrikama)"


if __name__ == "__main__":
    prefiksi = ucitaj_prefikse()
    llm = napravi_llm()

    # cisto uzorkovanje, isto kao u nasoj implementaciji
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                        max_tokens=MAX_NOVIH, min_tokens=MAX_NOVIH, seed=SEED)

    ulazi = [{"prompt_token_ids": ids} for ids in prefiksi]

    llm.generate(ulazi[:1], SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                                           max_tokens=8))          # zagrevanje
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    izlazi = llm.generate(ulazi, sp)
    torch.cuda.synchronize()
    trajanje = time.perf_counter() - t0

    tokena = sum(len(o.outputs[0].token_ids) for o in izlazi)
    oznaka = "baseline (bez spekulacije)" if GAMMA == 0 else f"spekulativno, gamma={GAMMA}"

    print()
    print("=" * 62)
    print(f"  {oznaka}")
    print(f"  target {TARGET_ID}" + (f" | draft {DRAFT_ID}" if GAMMA else ""))
    print(f"  {len(prefiksi)} prefiksa x {MAX_NOVIH} tokena")
    print("-" * 62)
    print(f"  vreme        : {trajanje:.2f} s")
    print(f"  tokena       : {tokena}")
    print(f"  propusnost   : {tokena / trajanje:.2f} tok/s")
    if GAMMA:
        print(f"  alfa (metrike): {prijavi_alfa(llm)}")
    print("=" * 62)
