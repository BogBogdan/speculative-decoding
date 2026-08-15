# Govor — slajdovi 1–8

Izgovoreni tekst je na engleskom, jer su slajdovi na engleskom.
Sitnim slovima su režijske napomene i klikovi.

---

## 01 · Naslov — *Speculative decoding* · ~30 s

> Good morning. My topic is speculative decoding, a way of making a large
> language model generate text faster.
>
> What makes it worth a thesis is the guarantee. The output is not approximated,
> not degraded, not "close enough". It is bit-for-bit identical to what the model
> would have produced anyway. We change how the work is scheduled, never what
> comes out of it.
>
> The setup is a fourteen-billion-parameter target model paired with a
> half-billion draft model, on Wikipedia text.

*Traka gore: tri prihvaćena tokena, jedan odbijen, dva koja čekaju. Ne objašnjavaj je
sada — vraća se na poslednjem slajdu, i tada su svi zeleni.*

---

## 02 · *The problem* · ~1 min · **2 klika**

> Let me start with a question. Have you ever wondered how a model with billions
> of parameters can answer you in real time?

*Pauza. Pusti da pitanje odstoji dve sekunde.*

> **[ENTER]** The honest answer is: it doesn't. It answers you one token at a time.
>
> **[ENTER]** Every token you see costs a full pass through all fourteen billion
> parameters. A five-hundred-token answer means five hundred passes. The model
> reads everything it knows, five hundred times over.
>
> And you cannot batch your way out of it, because each token depends on the one
> before it. That dependency is inherent to the method, not a flaw in our code.
>
> So the question this thesis asks is: can we make it faster while the output
> stays provably identical?

---

## 03 · *Guess ahead. Verify all at once.* · ~1 min 15 s · **2 klika**

> The idea is this. Instead of asking the big model for one token, we let a
> small, cheap model write several tokens ahead. Then the big model checks all of
> them in a single pass.
>
> **[ENTER]** Here is why that works. Writing is sequential — each guess depends
> on the one before it, so the draft has to go one at a time. But checking is
> not. The tokens are already there, so the target scores every position at once,
> in one forward pass instead of four.
>
> And it stops at the first rejection. Everything after it is thrown away.

*Pokaži rukom razliku: gore lanac strelica, dole snop paralelnih.*

> **[ENTER]** The idea is not mine. It comes from two papers, both from 2023. The
> first, from Google Research, introduces the algorithm, proves the output is
> unchanged, and gives the speedup model I will use throughout. The second, from
> DeepMind, derives the same procedure independently and confirms it at seventy
> billion parameters.
>
> I implemented Algorithm 1 from the first paper and benchmarked it against
> reference implementations.

---

## 04 · *Four steps, one round* · ~1 min · **4 klika, po jedan na kutiju**

> The procedure has four steps.
>
> **[ENTER]** First, guess. The draft writes gamma tokens ahead — sequentially,
> but cheaply, because the model is small.
>
> **[ENTER]** Second, verify. The target computes its own distribution for all
> those positions in a single pass. That bracket is the point: one call, every
> position.
>
> **[ENTER]** Third, decide. Each guess is accepted with probability min of one
> and p over q — p is what the target thinks, q is what the draft thought. We stop
> at the first rejection.
>
> **[ENTER]** Fourth, correct. We resample that rejected position from a corrected
> distribution. And if every guess passed, we get one extra token for free.
>
> Three tokens out of one call, in that example.

---

## 05 · *The draft samples from the wrong distribution* · ~1 min 10 s

> This is the part that makes the method interesting, so let me be precise.
>
> The draft samples from q, but we need a sample from p. Two rules turn one into
> the other.
>
> If the guess is accepted, we keep it — with probability min of one and p over q.
> Tokens the target likes more than the draft does always pass.
>
> If it is rejected, we do not simply resample from p. We resample from the
> leftover, p minus q, clipped at zero — only what the target wanted and the draft
> missed.
>
> And here is what that gives you.

*Pređi na tabelu. Pokazuj kolonu po kolonu.*

> The draft here is badly wrong. It gives token A a probability of 0.6; the target
> gives it 0.2. Yet look at the last column: 0.2, 0.5, 0.3 — exactly the target's
> own distribution.
>
> That holds for every token, every position, every pair of models. A poor draft
> only means slower. It never means different.

*Ako pitaju za dokaz: min(p,q) + (p−q)⁺ = p, jer je za svaki token tačno jedan od dva
člana različit od nule.*

---

## 06 · *Everything follows from two numbers* · ~50 s

> Everything about performance follows from two numbers.
>
> Gamma is how far ahead we guess. E is how many tokens come out of one target
> call. Those two are bookkeeping.
>
> The two that matter are alpha and c. Alpha is the acceptance rate — how often
> the target accepts a guess. It is a property of the pair of models, not of the
> code. And c is the cost ratio — what one draft step costs relative to one target
> step. That one is a property of the implementation and the hardware.
>
> Put together: speedup equals E over gamma-c plus one. The top is what a call
> gives us, the bottom is what it costs. There is a speedup only while the top
> grows faster than the bottom.
>
> That single line explains every result that follows.

---

## 07 · *The smallest one wins* · ~1 min

> The first question anyone asks is: why not use a bigger draft? So we measured
> three — half a billion, one and a half, and three billion parameters, all
> against the same target.
>
> The bigger drafts are better guessers. Alpha rises from 0.905 to 0.929 to 0.934.
>
> But look at the cost column: 0.044, then 0.138, then 0.276.
>
> And notice the shape of it. Going from half a billion to one and a half buys
> 0.024 in alpha. Going from one and a half to three buys only 0.005 — five times
> less. Quality saturates. Cost does not: every doubling is paid in full.
>
> So the smallest draft wins, and it wins at every gamma.

---

## 08 · *Speedup against guess depth* · ~40 s

> The same result as a picture, and it shows two things the table cannot.
>
> First, all three curves peak at gamma three. The optimal guess depth does not
> depend on the size of the draft — so that choice transfers.
>
> Second, the curves never cross. The small draft is better at every single
> depth, not just on average.
>
> And the gap widens as we go right: the deeper you guess, the more a large draft
> costs you.

---

**Zbirno vreme slajdova 1–8: ≈ 7 minuta.**

Ako te stisne vreme, skrati slajd 04 na dve rečenice — slajd 05 ionako ponavlja
odluku i ispravku detaljnije.
