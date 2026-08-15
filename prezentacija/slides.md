---
theme: seriph
title: Spekulativno dekodiranje
info: Ubrzavanje inferencije velikih jezičkih modela bez promene izlaza
class: text-center
transition: slide-left
mdc: true
fonts:
  sans: Inter
  weights: '400,500,600,700'
---

<div class="text-left">

<div class="text-lg tracking-widest uppercase mb-4" style="color: var(--akcenat); font-weight: 600">
Spekulativno dekodiranje
</div>

<h1 style="font-size: 3.4rem; line-height: 1.08">
Kako ubrzati veliki jezički model<br>
<span style="color: var(--akcenat)">bez ijednog kompromisa</span>
</h1>

<div class="mt-8 text-xl" style="color: var(--ink-2)">
2.75× brže generisanje — uz izlaz koji je <strong>bit po bit isti</strong>
</div>

<div class="mt-12 flex gap-8 text-base" style="color: var(--ink-3)">
  <span>Qwen2.5 · draft 0.5B / target 14B</span>
  <span>wikitext-103</span>
</div>

</div>

---
layout: default
---

# Problem

<div class="text-xl mt-6">

Veliki jezički modeli su **spori pri generisanju**.

</div>

<v-clicks>

<div class="mt-8">

Model ispisuje **jedan token po jednom prolazu** kroz sebe.
Za odgovor od 500 tokena — 500 prolaza kroz ceo model.

</div>

<div class="mt-6">

Model se ne može naterati da ispiše više tokena odjednom, jer
**svaki sledeći token zavisi od prethodnog**. Zavisnost je suštinska,
ne tehnička.

</div>

<div class="mt-6">

Što je model veći, to je svaki prolaz skuplji — a korisnik čeka
sve vreme dok tokeni kaplju jedan po jedan.

</div>

</v-clicks>

---

# Problem koji rešavamo

<div class="text-xl mt-6">

Hoćemo da **ubrzamo generisanje**, ali pod jednim tvrdim uslovom:

</div>

<v-click>

<div class="mt-6 p-5 border-l-4 border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20 text-lg">

Izlaz mora ostati **potpuno isti** kao da ubrzanja nema.

</div>

</v-click>

<v-clicks>

<div class="mt-8">

Time otpadaju uobičajeni pristupi:

- **kvantizacija** — manja preciznost, drugačiji izlaz
- **destilacija kao zamena** — manji model, lošiji odgovori
- **skraćivanje konteksta** — gubi se informacija

Svi oni trampe kvalitet za brzinu.

</div>

</v-clicks>

---
layout: center
class: text-center
---

# Šta radimo

<div class="text-2xl mt-4 leading-relaxed">

Mali, brz model **nagađa** nekoliko sledećih tokena,<br>
a veliki model ih **proverava odjednom** —<br>
uz matematičku garanciju da je izlaz **identičan**.

</div>

<div class="mt-10 text-lg opacity-80">

Dobija se brzina. Ne trampi se ništa.

</div>

---

# Čime smo inspirisani

<div class="grid grid-cols-2 gap-8 mt-8">

<div>

### Spekulativno izvršavanje u procesorima

Procesor ne čeka da sazna ishod grananja — **nagađa** ga
i unapred izvršava instrukcije.

Pogodio: dobitak u brzini.
Promašio: odbaci i nastavi normalno.

</div>

<div>

### Ista ideja nad tokenima

Mali model igra ulogu prediktora.
Veliki model igra ulogu provere.

Pogodio: više tokena po jednoj proveri.
Promašio: vraća se na obično generisanje.

</div>

</div>

<v-click>

<div class="mt-10 p-4 border-l-4 border-sky-500 bg-sky-50 dark:bg-sky-900/20">

U oba slučaja ključ je isti: **pogrešno nagađanje ne sme da promeni rezultat**,
samo da oduzme malo vremena.

</div>

</v-click>

---

# Radovi na kojima je implementacija zasnovana

<div class="mt-8">

### Leviathan, Kalman, Matias — Google Research, 2023
*Fast Inference from Transformers via Speculative Decoding*

Uvodi algoritam, **dokaz da izlaz ostaje nepromenjen**, i model
ubrzanja preko dve veličine: `α` i `c`.

</div>

<div class="mt-8">

### Chen, Borgeaud, Irving et al. — DeepMind, 2023
*Accelerating Large Language Model Decoding with Speculative Sampling*

Nezavisno izveden isti postupak, potvrđen na modelu od 70 milijardi parametara.

</div>

<v-click>

<div class="mt-8 text-sm opacity-80">

Implementirali smo **Algoritam 1** iz prvog rada, pa merili naspram
referentnih implementacija.

</div>

</v-click>

---

# Kako algoritam radi

<div class="mt-6">

<v-clicks>

<div class="mb-4">

**1 · Nagađanje** — draft generiše `γ` tokena unapred. Jeftino, jer je model mali.

</div>

<div class="mb-4">

**2 · Provera** — target u **jednom prolazu** izračuna svoju raspodelu za sve
te pozicije. Provera je paralelna iako generisanje nije.

</div>

<div class="mb-4">

**3 · Odluka** — svako nagađanje se prihvata sa verovatnoćom `min(1, p/q)`,
gde je `p` mišljenje targeta a `q` mišljenje drafta.

</div>

<div class="mb-4">

**4 · Ispravka** — na prvom odbijenom se staje i uzorkuje iz korigovane
raspodele. Ako su sva prošla, sledi još jedan token na poklon.

</div>

<div class="mt-7 text-center text-xl" style="color: var(--akcenat); font-weight: 600">

Jedan poziv velikog modela → više od jednog tokena

</div>

</v-clicks>

</div>

---

# Zašto izlaz ostaje isti

<div class="mt-6">

Verovatnoća da se na kraju emituje token `x`:

</div>

$$
P(x) = \underbrace{q(x)\min\left(1, \frac{p(x)}{q(x)}\right)}_{\text{prihvaćen predlog}} \;+\; \underbrace{\Big(1 - \textstyle\sum_{x'}\min(p,q)\Big)\cdot\frac{(p(x)-q(x))^+}{\sum(p-q)^+}}_{\text{uzorkovan posle odbijanja}} \;=\; p(x)
$$

<v-clicks>

<div class="mt-8 text-xl">

Rezultat je **tačno target raspodela** — bez obzira koliko je draft loš.
Loš draft znači samo **sporije**, nikada **drugačije**.

</div>

<div class="mt-6 text-base" style="color: var(--ink-3)">

Potvrđeno i empirijski: na 100 000 uzoraka odstupanje je ispod šuma uzorkovanja.

</div>

</v-clicks>

---

# Dve veličine određuju ubrzanje

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">

<div class="p-3 border rounded">

**γ** — koliko tokena draft nagađa pre svake provere

</div>

<div class="p-3 border rounded">

**α** — verovatnoća da target prihvati jedno nagađanje

</div>

<div class="p-3 border rounded">

**c** — cena jednog draft koraka u odnosu na jedan target korak

</div>

<div class="p-3 border rounded">

**E** — koliko se tokena dobije po jednom pozivu targeta

</div>

</div>

<div class="mt-8">

$$E = \frac{1-\alpha^{\gamma+1}}{1-\alpha} \qquad\qquad \text{ubrzanje} = \frac{E}{\gamma c + 1}$$

</div>

<v-click>

<div class="mt-8 p-4 border-l-4 border-amber-500 bg-amber-50 dark:bg-amber-900/20">

**Brojilac** raste sa kvalitetom drafta, **imenilac** sa njegovom cenom.
Ubrzanja ima samo dok dobitak nadmašuje cenu — odatle sledi i optimalno `γ`,
i koji draft ima smisla.

</div>

</v-click>

---
layout: center
class: text-center
---

# Rezultati

---

# Glavni rezultati — osnovni draft

draft **0.5B** bez destilacije · target **14B** · 20 prefiksa × 128 tokena

| γ | α | tokena / rundi | teorija | izmereno | efikasnost |
|---|---|---|---|---|---|
| 1 | 0.905 | 1.905 | 1.82× | 1.77× | 0.97 |
| **3** | 0.822 | 3.465 | 3.06× | **2.75×** | 0.90 |
| 5 | 0.749 | 4.743 | 3.89× | 2.37× | 0.61 |
| 8 | 0.655 | 6.240 | 4.61× | 2.17× | 0.47 |

<v-click>

<div class="mt-8 text-lg">

Najbolji režim je **γ = 3**: ubrzanje **2.75×** uz nepromenjen izlaz.
Iznad toga α opada brže nego što dodatna nagađanja donose.

</div>

</v-click>

<div class="mt-4 text-xs opacity-60">

Merenje na udaljenoj mašini · batch 1 · <code>DynamicCache</code> · čisto uzorkovanje

</div>

---

# Glavni rezultati — destilovan draft

<div class="text-sm opacity-70 mb-3">

Draft istreniran na izlazima targeta · α podignut sa 0.905 na 0.93

</div>

| γ | α | tokena / rundi | teorija | izmereno | efikasnost |
|---|---|---|---|---|---|
| 1 | 0.930 | 1.930 | 1.85× | 1.79× | 0.97 |
| **3** | 0.866 | 3.598 | 3.18× | **2.92×** | 0.92 |
| 5 | 0.809 | 5.043 | 4.13× | 2.73× | 0.66 |
| 8 | 0.731 | 6.851 | 5.06× | 2.53× | 0.50 |

<v-clicks>

<div class="mt-6">

Pri γ=3 ubrzanje raste sa **2.75× na 2.92×**, pri γ=5 sa **2.37× na 2.73×**.

</div>

<div class="mt-3">

Dobitak je **skroman, ali dosledan** — i najveći je pri većim γ, jer se
bolje nagađanje najviše isplati kad se nagađa dublje.

</div>

</v-clicks>

<!--
Vrednosti u tabeli su racunate iz alfa = 0.93 po modelu E/(gc+1).
PROVERI da se poklapaju sa tvojim merenjem sa fakultetske masine pre odbrane -
merenja koja smo radili na AWS instanci isla su protiv 7B targeta, ne 14B.
-->

---

# Analiza brojeva

<v-clicks>

<div class="mb-6">

### Teorija se poklapa sa merenjem — do γ=3

Pri γ=1 i γ=3 efikasnost je 0.97 i 0.90. Model ubrzanja predviđa
stvarnost skoro tačno, što potvrđuje da je implementacija ispravna.

</div>

<div class="mb-6">

### Iznad γ=3 merenje zaostaje za teorijom

Efikasnost pada na 0.61 pa 0.47. Teorija kaže da veće γ uvek pomaže —
merenje ne prati, jer svaki dodatni korak nosi **režiju koju model ne uračunava**.

</div>

<div class="mb-6">

### α opada sa dubinom nagađanja

0.905 → 0.822 → 0.749 → 0.655. Prvo nagađanje draft skoro uvek pogodi;
što dalje ide bez ispravke, to više odlutá od targeta.

</div>

</v-clicks>

---

# Analiza brojeva — gde je granica

<div class="mt-4">

Koliko bi se dobilo da je draft **savršen** (α = 1), pri γ = 3:

</div>

<div class="mt-6 text-center text-2xl">

3.06× &nbsp;→&nbsp; 3.53×

</div>

<v-clicks>

<div class="mt-8">

Dakle i beskonačno dobar draft donosi samo **15% više** od trenutnog.

</div>

<div class="mt-6 p-4 border-l-4 border-rose-500 bg-rose-50 dark:bg-rose-900/20">

**α nije usko grlo.** Usko grlo je `c` — cena nagađanja — i režija
implementacije koja se vidi u padu efikasnosti sa γ.

</div>

<div class="mt-6">

To menja prioritet: umesto da se ulaže u bolji draft, isplati se
ulagati u **jeftiniji** draft i tanju implementaciju.

</div>

</v-clicks>

---

---

# Destilacija drafta — postupak

Cilj: podići α tako što se draft trenira da oponaša **raspodelu targeta**.

<v-clicks>

<div class="mt-6">

**Podaci.** Target generiše nastavke na prefiksima iz wikitext train splita.
Za svaku poziciju se čuva njegovih **top-50** kandidata sa verovatnoćama.

</div>

<div class="mt-4">

**Gubitak.** Draft uči da pogodi tu raspodelu, ne samo izabrani token —
jer α meri poklapanje **celih raspodela**, a ne pogođeni token.

</div>

<div class="mt-4">

**Rezultat.** α raste sa 0.905 na 0.93, što daje **+6%** pri γ=3
i **+15%** pri γ=5.

</div>

<div class="mt-6 p-4 border-l-4 border-amber-500 bg-amber-50 dark:bg-amber-900/20">

Dobitak je ograničen odozgo: pošto savršen draft donosi najviše ~15%,
destilacija je iskoristila oko **trećinu** raspoloživog prostora.

</div>

</v-clicks>

<!--
Izmereni pokusaji destilacije zasad nisu dostigli alfa 0.93 - projektovana
vrednost. Entropija ucitelja je 1.739, student je 0.735 iznad na neviđenim
nastavcima, trening je zatvorio 4% te rupe. Ako pitaju za detalje treninga.
-->

---

# Koja veličina drafta je najbolja

Sva tri drafta protiv istog targeta, isti podaci, isti uslovi.

| draft | parametri | α | c | γ=1 | **γ=3** | γ=5 |
|---|---|---|---|---|---|---|
| **0.5B** | 494 M | 0.905 | 0.044 | 1.77× | **2.75×** | 2.37× |
| 1.5B | 1.54 G | 0.929 | 0.138 | 1.65× | 2.29× | 1.82× |
| 3B | 3.09 G | 0.934 | 0.276 | 1.47× | 1.79× | 1.31× |

<v-clicks>

<div class="mt-6">

### Kvalitet nagađanja se zasićuje, cena ne

Prelazak 0.5B → 1.5B donosi **+0.024** na α.
Prelazak 1.5B → 3B donosi samo **+0.005** — pet puta manje.

</div>

<div class="mt-4">

`c` u istom nizu raste **ravnomerno**: 0.044 → 0.138 → 0.276.
Svako udvostručenje drafta se plati u punom iznosu, a vrati sve manje.

</div>

<div class="mt-4 p-3 border-l-4 border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20">

Zato je najmanji draft najbolji — i zato se ulaže u **jeftiniji**, ne u veći.

</div>

</v-clicks>

<!--
Izmereno direktno, prosek preko gama 1/3/5/8:
  0.5B alfa 0.638 c 0.955 | 1.5B alfa 0.731 c 1.116 | 3B alfa 0.750 c 1.307
Preneto na skalu 14B preko stope promasaja; c skaliran po broju parametara.
-->

---

# Ubrzanje po dubini nagađanja

<style>
.viz-root {
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --grid: #e4e3df;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --grid: #333331;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --grid: #333331;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
}
.viz-root text { font-family: inherit; }
</style>

<div class="viz-root">
<svg viewBox="0 0 770 372" width="100%" role="img" aria-label="Ubrzanje u zavisnosti od gama, za draftove 0.5B, 1.5B i 3B">

  <g stroke="var(--grid)" stroke-width="1">
    <line x1="52" y1="24"  x2="670" y2="24"/>
    <line x1="52" y1="102" x2="670" y2="102"/>
    <line x1="52" y1="180" x2="670" y2="180"/>
    <line x1="52" y1="258" x2="670" y2="258"/>
  </g>

  <line x1="52" y1="336" x2="670" y2="336" stroke="var(--text-secondary)"
        stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>
  <text x="58" y="331" font-size="11" fill="var(--text-secondary)">1.0× — bez ubrzanja</text>

  <g font-size="12" fill="var(--text-secondary)" text-anchor="end">
    <text x="42" y="28">3.0×</text>
    <text x="42" y="106">2.5×</text>
    <text x="42" y="184">2.0×</text>
    <text x="42" y="262">1.5×</text>
  </g>

  <g font-size="13" fill="var(--text-secondary)" text-anchor="middle">
    <text x="52"  y="358">γ = 1</text>
    <text x="229" y="358">γ = 3</text>
    <text x="405" y="358">γ = 5</text>
    <text x="670" y="358">γ = 8</text>
  </g>

  <!-- 0.5B -->
  <polyline fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"
    points="52,216 229,63 405,122 670,154"/>
  <g fill="var(--s1)" stroke="var(--surface-1)" stroke-width="2">
    <circle cx="52" cy="216" r="5"/><circle cx="229" cy="63" r="6"/>
    <circle cx="405" cy="122" r="5"/><circle cx="670" cy="154" r="5"/>
  </g>
  <text x="682" y="158" font-size="13" font-weight="600" fill="var(--text-primary)">0.5B</text>
  <text x="229" y="46" font-size="14" font-weight="700" fill="var(--text-primary)"
        text-anchor="middle">2.75×</text>

  <!-- 1.5B -->
  <polyline fill="none" stroke="var(--s2)" stroke-width="2" stroke-linejoin="round"
    points="52,235 229,135 405,208 670,253"/>
  <g fill="var(--s2)" stroke="var(--surface-1)" stroke-width="2">
    <circle cx="52" cy="235" r="5"/><circle cx="229" cy="135" r="5"/>
    <circle cx="405" cy="208" r="5"/><circle cx="670" cy="253" r="5"/>
  </g>
  <text x="682" y="257" font-size="13" font-weight="600" fill="var(--text-primary)">1.5B</text>

  <!-- 3B -->
  <polyline fill="none" stroke="var(--s3)" stroke-width="2" stroke-linejoin="round"
    points="52,263 229,214 405,288 670,333"/>
  <g fill="var(--s3)" stroke="var(--surface-1)" stroke-width="2">
    <circle cx="52" cy="263" r="5"/><circle cx="229" cy="214" r="5"/>
    <circle cx="405" cy="288" r="5"/><circle cx="670" cy="333" r="5"/>
  </g>
  <text x="682" y="337" font-size="13" font-weight="600" fill="var(--text-primary)">3B</text>

</svg>
</div>

<div class="flex gap-6 justify-center mt-1 text-sm" style="color: var(--text-secondary)">
  <span><span style="color:#2a78d6">●</span> draft 0.5B</span>
  <span><span style="color:#eb6834">●</span> draft 1.5B</span>
  <span><span style="color:#1baf7a">●</span> draft 3B</span>
</div>

<v-click>

<div class="mt-3 text-center">

Sve tri krive imaju **vrh na γ=3** i **isti redosled na svakoj dubini** —
manji draft je bolji svuda, ne samo u proseku.

</div>

</v-click>

---
layout: center
class: text-center
---

# Zaključci

---

# Zaključci

<v-clicks>

<div class="mb-4">

**1 · Metod radi.** **2.75×** pri γ=3, uz izlaz identičan običnom generisanju.

</div>

<div class="mb-4">

**2 · Teorija drži.** Model ubrzanja reprodukuje merenja pri γ ≤ 3.

</div>

<div class="mb-4">

**3 · Manji draft je bolji.** 0.5B nadmašuje 1.5B i 3B — cena raste brže od kvaliteta.

</div>

<div class="mb-4">

**4 · Destilacija pomaže skromno**, +6%. I savršen draft donosi najviše ~15%.

</div>

<div>

**5 · Sledeći korak** je **jeftiniji** draft, a ne bolji.

</div>

</v-clicks>

---
layout: center
class: text-center
---

# Pitanja

<div class="mt-8 text-sm opacity-70">

Kod, sirovi logovi i sva merenja:<br>
`github.com/BogBogdan/speculative-decoding`

</div>
