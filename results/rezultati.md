# Rezultati merenja

Sva merenja spekulativnog dekodiranja, sa punom konfiguracijom uz svaki broj.

**Hardver.** AWS `g5.xlarge`, NVIDIA A10G 22.5 GB, 4 vCPU, 15 GB RAM, eu-north-1.
**Softver.** Python 3.14, torch 2.13.0+cu130, transformers 5.15.0, vLLM 0.27.1, CUDA drajver 595.x, nvcc 12.4.
**Modeli.** draft `Qwen/Qwen2.5-0.5B`, target `Qwen/Qwen2.5-7B`, oba u bf16.
**Prefiksi.** `data/prefiksi.pt` — 100 pasusa iz wikitext-103-raw-v1 test splita, svaki odsečen na tačno 50 tokena.
**Uzorkovanje.** Čisto, `temperature=1.0, top_k=0, top_p=1.0`, bez top-k/top-p filtriranja.
**Rečnik.** Logiti odsečeni na `len(tokenizer) = 151665` pre `softmax`-a (0.5B ima 151936, 7B 152064).

Merenja su rađena na dve instance. Sirovi logovi druge su u `results/sirovi-logovi/`;
prva je terminirana, pa su njeni brojevi prepisani iz sesije.

---

## 1. Pun eksperiment — glavni brojevi

`scripts/run_eval.py`, `DynamicCache`, batch 1, **20 prefiksa × 128 tokena**, 964 iteracije,
`GEMMA=5 MERI_BASELINE=1`. Instanca 1.

| veličina | vrednost |
|---|---|
| **α (acceptance rate)** | **0.6620** |
| **c (cost coefficient)** | **0.9065** — draft 31.3 ms, target 34.5 ms |
| tokena / poziv target modela | 2.6805 (teorijski 2.7098) |
| teorijsko ubrzanje | 0.4898× |
| **izmereno ubrzanje** | **0.6277×** — spec 143.5 s, baseline 90.1 s |
| raspodela n | `[361, 193, 126, 98, 49, 137]` |
| perplexity draft / target | 28.46 / 15.63 |

---

## 2. Pretraga po γ — ova implementacija

`scripts/run_eval.py`, `DynamicCache`, batch 1, **5 prefiksa × 64 tokena**, `MERI_BASELINE=1`. Instanca 1.

| γ | α | c | tokena/poziv | teorijski | teorijsko ubrz. | izmereno |
|---|---|---|---|---|---|---|
| 1 | 0.643 | 0.947 | 1.643 | 1.643 | 0.844× | **0.905×** |
| 2 | 0.636 | 0.944 | 2.031 | 2.040 | 0.706× | 0.808× |
| 3 | 0.660 | 0.941 | 2.348 | 2.382 | 0.623× | 0.745× |
| 5 | 0.618 | 0.953 | 2.470 | 2.470 | 0.428× | 0.539× |
| 8 | 0.643 | 0.958 | 2.713 | 2.749 | 0.317× | 0.404× |

---

## 3. Poređenje keševa

Instanca 2, **5 prefiksa × 64 tokena**, `MERI_BASELINE=1`.
Log: `sirovi-logovi/uporedi.log`.

| γ | varijanta | α | c | tokena/poziv | izmereno |
|---|---|---|---|---|---|
| 1 | `run_eval.py`, `DynamicCache` | 0.5891 | 1.1246 | 1.5891 (1.5891) | 0.9076× |
| 1 | `run_eval_static.py`, `COMPILE=0` | 0.6131 | — | 1.6131 (1.6131) | 0.6715× |
| 1 | `run_eval_static.py`, `COMPILE=1` | — | — | — | **pao** |
| 5 | `run_eval.py`, `DynamicCache` | 0.6369 | 1.0049 | 2.5385 (2.5705) | 0.5909× |
| 5 | `run_eval_static.py`, `COMPILE=0` | 0.6154 | — | 2.4222 (2.4588) | 0.4372× |
| 5 | `run_eval_static.py`, `COMPILE=1` | — | — | — | **pao** |

`COMPILE=1` puca u `cache_utils.py:476` — `StaticCache` menja `cumulative_length`
na mestu unutar kompajlirane oblasti, što CUDA grafovi ne podnose.

---

## 4. `torch.compile` — c pada, izmereno se ruši

Instanca 1, **5 prefiksa × 64 tokena**, `MERI_BASELINE=1`, `DynamicCache`.

| varijanta | γ | draft | target | c | teorijsko | izmereno |
|---|---|---|---|---|---|---|
| bez compile | 1 | 32.6 ms | 34.4 ms | 0.947 | 0.844× | 0.905× |
| `torch.compile` | 1 | 7.9 ms | 32.0 ms | 0.246 | 1.275× | 0.070× |
| `torch.compile` | 5 | 7.9 ms | 32.0 ms | 0.246 | 1.089× | 0.066× |
| `dynamic=True` | 1 | 8.7 ms | 32.0 ms | 0.273 | 1.300× | 0.058× |
| `dynamic=True` | 5 | 8.5 ms | 32.0 ms | 0.265 | 1.048× | 0.081× |

Kompilacija spusti `c` ispod praga, ali stalna rekompilacija u spekulativnoj petlji
pojede sve — oko 960 ms po iteraciji mimo poziva modela.

---

## 5. HF assisted generation — referentna implementacija

`target.generate(assistant_model=draft)`, instanca 1, batch 1, **5 prefiksa × 128 tokena**.
Oba modela prethodno svedena na 151665 preko `resize_token_embeddings`.

| konfiguracija | propusnost | ubrzanje |
|---|---|---|
| obično `generate` | 28.17 tok/s | 1.000× |
| assisted, γ=1 | 23.67 tok/s | 0.817× |
| assisted, γ=3 | 21.90 tok/s | 0.726× |
| assisted, γ=5 | 19.23 tok/s | 0.638× |
| assisted, γ=8 | 18.89 tok/s | 0.666× |

Bez odsecanja rečnika odbija sa `ValueError: The main and assistant models have different tokenizers`.

---

## 6. Prompt lookup — draft bez modela

`target.generate(prompt_lookup_num_tokens=γ)`, instanca 2, batch 1,
**20 prefiksa × 128 tokena**. Log: `sirovi-logovi/lookup.log`.

| konfiguracija | vreme | propusnost | ubrzanje |
|---|---|---|---|
| obično `generate` | 89.72 s | 28.53 tok/s | 1.000× |
| lookup, γ=2 | 94.72 s | 27.03 tok/s | 0.947× |
| lookup, γ=3 | 92.73 s | 27.61 tok/s | 0.967× |
| lookup, γ=5 | 94.29 s | 27.15 tok/s | 0.952× |
| lookup, γ=8 | 96.08 s | 26.64 tok/s | 0.934× |
| lookup, γ=10 | 96.58 s | 26.51 tok/s | 0.929× |

α je praktično nula — n-grami se na nastavku wikitext pasusa ne ponavljaju dovoljno.
Ostaje samo režija HF-ovog assisted puta.

---

## 7. vLLM — jedina varijanta preko 1×

`scripts/run_eval_vllm.py`, vLLM 0.27.1, instanca 2, **20 prefiksa × 128 tokena**,
`gpu_memory_utilization=0.85`, modeli iz `~/draft-151665` i `~/target-151665`.

### batch 1 (`PO_JEDAN=1`) — kašnjenje

Logovi: `sirovi-logovi/b1_0.txt`, `b1_1.txt`, `b1_3.txt`, `b1_5.txt`.

| γ | vreme | propusnost | ubrzanje | izvedeno c |
|---|---|---|---|---|
| 0 (baseline) | 83.11 s | 30.80 tok/s | 1.000× | — |
| **1** | **68.55 s** | **37.34 tok/s** | **1.212×** | 0.369 |
| **3** | **68.89 s** | **37.16 tok/s** | **1.206×** | 0.327 |
| **5** | **75.14 s** | **34.07 tok/s** | **1.106×** | 0.290 |

`izvedeno c` je iz formule `ubrzanje = E[tokena] / (γ·c + 1)` uz α = 0.66.
Sve tri vrednosti su bliske, što potvrđuje da model iz rada drži.

### batch 20 (podrazumevano) — propusnost

Log: `sirovi-logovi/vllm4.log`.

| γ | vreme | propusnost | ubrzanje |
|---|---|---|---|
| 0 (baseline) | 4.69 s | 545.87 tok/s | 1.000× |
| 1 | 4.35 s | 589.06 tok/s | 1.079× |
| 3 | 5.75 s | 445.11 tok/s | 0.815× |
| 5 | 6.23 s | 410.97 tok/s | 0.753× |
| 8 | 8.38 s | 305.40 tok/s | 0.559× |

Pri batch 20 kartica je zasićena, pa nagađanja oduzimaju kapacitet umesto da popunjavaju prazan hod.

---

## 7b. vLLM `ngram` — draft bez modela

`scripts/run_eval_vllm.py` sa `METODA=ngram`, `prompt_lookup_min=2`, `prompt_lookup_max=4`.
Nagađanja se traže u već viđenom tekstu, bez ijednog poziva modela — dakle `c ≈ 0`.
Instanca 2, **20 prefiksa × 128 tokena**.
Logovi: `sirovi-logovi/n1_ngram_*.txt` (batch 1), `ng_ngram_*.txt` (batch 20).

### batch 1

| metoda | γ | vreme | propusnost | ubrzanje |
|---|---|---|---|---|
| baseline | — | 83.11 s | 30.80 tok/s | 1.000× |
| `ngram` | 3 | 85.13 s | 30.07 tok/s | 0.976× |
| `ngram` | 5 | 85.43 s | 29.97 tok/s | 0.973× |
| `ngram` | 8 | 85.45 s | 29.96 tok/s | 0.973× |

### batch 20

| metoda | γ | propusnost | ubrzanje |
|---|---|---|---|
| baseline | — | 588.12 tok/s | 1.000× |
| `ngram` | 3 | 509.75 tok/s | 0.867× |
| `ngram` | 5 | 506.36 tok/s | 0.861× |
| `ngram` | 8 | 506.33 tok/s | 0.861× |
| `ngram_gpu` | 5 | 458.29 tok/s | 0.779× |

**α je praktično nula.** Sva tri γ daju isti broj do treće decimale — kad povećanje broja
nagađanja ništa ne menja, znači da se ništa ne prihvata. Prefiks je 50 tokena, model dopisuje
128 novih, i u tako kratkom rasponu se n-grami enciklopedijske proze ne ponavljaju.
Metoda je namenjena kodu, JSON-u i zadacima gde se ulaz prepisuje u izlaz.

`suffix` metoda nije merena — traži `arctic-inference`, koji se ne gradi na ovoj mašini.

Isti zaključak dao je i HF `prompt_lookup` (sekcija 6): besplatan draft ne pomaže ako nema šta da pogodi.
Formula `c = 0 ⇒ ubrzanje = E[tokena] ≥ 1` važi samo dok je i **režija** nagađanja nula, što ovde nije.

---

## 8. Provere ispravnosti

Sitni nasumični modeli (2 sloja, `vocab_size=64`) na CPU-u, `scratchpad/test_alg1.py`
i `test_static.py`. Ne zavise od hardvera.

| provera | očekivano | izmereno |
|---|---|---|
| tokena/poziv naspram `(1−α^(γ+1))/(1−α)` | poklapanje | 4 decimale, svih 5 γ |
| Theorem 1, Monte Carlo kroz modele (`DynamicCache`) | TV ≈ kontrola | 0.03602 / 0.03798 |
| Theorem 1, Monte Carlo kroz modele (`StaticCache`) | TV ≈ kontrola | 0.04444 / 0.03962 |
| Theorem 1, Monte Carlo nad raspodelama | TV ≈ kontrola | 0.05474 / 0.05568 |
| `draft == target` ⇒ α = 1 | 1.0000 | 1.0000 |
| bonus token: tokena/poziv pri α=1 | γ + 1 | tačno γ + 1 |
| rollback `DynamicCache.crop(-k)` | ≈ 0 | 8.94e-08 |
| rollback `StaticCache.cumulative_length.fill_()` | ≈ 0 | 5.96e-08 |
| oblici ulaza uz `StaticCache` | draft {1,2}, target γ+1 | potvrđeno |

---

## 9. Razlaganje c

Teorijsko vreme = težine u bf16 podeljeno propusnim opsegom A10G (~600 GB/s).

| model | težine | teorijski | HF eager | režija | `torch.compile` |
|---|---|---|---|---|---|
| draft 0.5B | 1.0 GB | 1.7 ms | 32.6 ms | **18×** | 7.9 ms |
| target 7B | 15.2 GB | 25.3 ms | 34.4 ms | 1.4× | 32.0 ms |

Target je memorijski ograničen — pretpostavka rada o njemu drži.
Draft nije: kartica ga završi za 1.7 ms pa čeka CPU da izda ~400 naredbi po tokenu.

Pragovi za ubrzanje preko 1× pri α = 0.66:

| γ | E[tokena] | potreban c |
|---|---|---|
| 1 | 1.662 | < 0.662 |
| 2 | 2.100 | < 0.550 |
| 3 | 2.390 | < 0.463 |
| 5 | 2.710 | < 0.342 |

Pri `c = 0.95` čak i savršen draft (α = 1) daje najviše **1.05×** — destilacija to ne može popraviti.

---

## 9b. Metrike pogađanja u najboljem slučaju

α je svojstvo **para modela**, ne implementacije, pa vrednost izmerena na 964 iteracije
(`run_eval.py`, γ=5) važi i za vLLM slučaj koji je dao 1.212×.

### Koliko se pogađa

```
α = 0.6620        dva od tri nagadjanja prolaze
```

### Gde staje — raspodela n

| prihvaćeno n | koliko puta | udeo | teorijski `α^n(1−α)` |
|---|---|---|---|
| 0 | 361 | 37.4% | 33.8% |
| 1 | 193 | 20.0% | 22.4% |
| 2 | 126 | 13.1% | 14.8% |
| 3 | 98 | 10.2% | 9.8% |
| 4 | 49 | 5.1% | 6.5% |
| 5 (svih) | 137 | 14.2% | 12.7% |

Poklapa se sa geometrijskom raspodelom koju teorija predviđa.

- U **37%** iteracija prvo nagađanje odmah pada — najčešći pojedinačni ishod
- U **14%** prođe svih pet, pa se dobija i bonus token
- Prosečno prihvaćeno: **1.68** tokena, plus jedan iz korekcije ili bonusa = **2.68**

Očekivan broj uzastopnih pogodaka pre prekida, bez ograničenja γ:

```
α / (1 − α) = 0.662 / 0.338 ≈ 1.96
```

Draft pogodi oko **dva tokena zaredom** pa promaši. Zato γ=1 i γ=3 pod vLLM-om daju
skoro isto ubrzanje (1.212× i 1.206×) — iznad tri nagađanja retko šta stigne da se iskoristi.

### Razlika draft naspram target

| po čemu | draft 0.5B | target 7B | odnos |
|---|---|---|---|
| parametri | 494 M | 7.62 G | 15.4× |
| težine bf16 | 1.0 GB | 15.2 GB | 15.2× |
| perplexity (isti prefiksi) | 28.46 | 15.63 | 1.82× lošiji |
| korak, teorijski | 1.7 ms | 25.3 ms | **c = 0.067** |
| korak, vLLM (izvedeno) | — | — | **c ≈ 0.30** |
| korak, HF eager | 32.6 ms | 34.4 ms | **c ≈ 0.95** |

### Koji od dva ograničava

| poluga | trenutno | granica | koliko fali |
|---|---|---|---|
| α | 0.662 | 1.0 | 1.5× |
| c | 0.30 | 0.067 | **4.5×** |

**α je u redu za ovaj par modela** — 0.66 je očekivano za jaz od 15× bez ikakve destilacije,
i ne može preći 1.0. **c je problem**: hardver dozvoljava 0.067, a dobija se 0.30, dakle
4.5× se gubi na režiji koja nema veze sa modelima.

Šta bi svaka poluga donela pri γ=5:

```
sadasnje stanje    α=0.662  c=0.30    ->  1.11x   (izmereno 1.106x)
savrsen draft      α=1.0    c=0.30    ->  2.40x
savrsena impl.     α=0.662  c=0.067   ->  2.03x
oboje              α=1.0    c=0.067   ->  4.49x
```

Destilacija i optimizacija implementacije su otprilike jednako vredne pri γ=5, ali je
`c` jedini koji se popravlja bez treniranja — i jedini gde se gubi nešto što hardver već ima.

---

## 10. Rečnik u Qwen2.5 porodici

| | `vocab_size` | višak preko tokenizera |
|---|---|---|
| tokenizer | 151665 | — |
| 0.5B / 1.5B / 3B | 151936 | 271 |
| 7B / 14B / 32B | 152064 | 399 |

Izmereno na 0.5B, jedan pravi prefiks:

```
masa verovatnoce na dopuni [151665..151936) = 1.214e-09
najveca pojedinacna                          = 4.488e-12
najverovatniji pravi token                   = 0.1071
dekodiranje id-a 151800                      = ''
eos_token_id = 151643 < 151665               ✓ nije odseceno
najveci specijalni token = 151656 < 151665   ✓ nije odseceno
```

Odbijaju par bez izjednačenog rečnika: `transformers` (assisted generation) i vLLM (`SpeculativeConfig`).
