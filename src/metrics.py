import math
import time
import torch
import torch.nn.functional as F


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    
def alfa(prihvaceno, ponudjeno):
    return prihvaceno / ponudjeno

@torch.no_grad()
def izmeri_c(draft, target, ids, ponavljanja=10):
    device = ids.device

    draft(ids)
    target(ids)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(ponavljanja):
        draft(ids)
    _sync(device)
    t_draft = (time.perf_counter() - t0) / ponavljanja

    t0 = time.perf_counter()
    for _ in range(ponavljanja):
        target(ids)
    _sync(device)
    t_target = (time.perf_counter() - t0) / ponavljanja

    return t_draft / t_target, t_draft, t_target

def ocekivano_tokena(a, gamma):
    if a >= 1.0:
        return float(gamma + 1)
    return (1.0 - a ** (gamma + 1)) / (1.0 - a)


def teorijsko_ubrzanje(a, gamma, c):
    return ocekivano_tokena(a, gamma) / (gamma * c + 1.0)


def izmereno_ubrzanje(t_baseline, t_spekulativno):
    return t_baseline / t_spekulativno


@torch.no_grad()
def autoregresivno(model, ids, max_novih):
    for _ in range(max_novih):
        p = torch.softmax(model(ids).logits[0, -1].float(), dim=-1)
        x = torch.multinomial(p, 1)
        ids = torch.cat([ids, x.unsqueeze(0)], dim=1)
    return ids

def odstupanje_raspodele(p, q):
    P = torch.minimum(p, q) + (p - q).clamp(min=0)
    return (P - p).abs().max().item()


def empirijska_ekvivalentnost(p, q, uzoraka=100000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    x = torch.multinomial(q, uzoraka, replacement=True, generator=g)
    odnos = (p[x] / q[x]).clamp(max=1.0)
    prihvaceno = torch.rand(uzoraka, generator=g) < odnos

    ostatak = (p - q).clamp(min=0)
    ostatak = ostatak / ostatak.sum()
    zamena = torch.multinomial(ostatak, uzoraka, replacement=True, generator=g)

    def tv(uzorci):
        hist = torch.bincount(uzorci, minlength=p.numel()).float() / uzoraka
        return 0.5 * (hist - p).abs().sum().item()

    spekulativno = torch.where(prihvaceno, x, zamena)
    kontrola = torch.multinomial(p, uzoraka, replacement=True, generator=g)
    return tv(spekulativno), tv(kontrola)


@torch.no_grad()
def perplexity(model, sekvence):
    device = next(model.parameters()).device
    ukupno_nll = 0.0
    ukupno_tokena = 0

    for ids in sekvence:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device)
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits[0, :-1].float()
        ciljevi = ids[0, 1:]
        ukupno_nll += F.cross_entropy(logits, ciljevi, reduction="sum").item()
        ukupno_tokena += ciljevi.numel()

    return math.exp(ukupno_nll / ukupno_tokena)


class Merenja:
    def __init__(self, gamma):
        self.gamma = gamma
        self.iteracija = 0
        self.prihvaceno = 0
        self.provereno = 0
        self.tokena = 0
        self.vreme = 0.0
        self.max_odstupanje = 0.0
        self.histogram_n = [0] * (gamma + 1) 

    def dodaj(self, n, odstupanje=0.0):
        self.iteracija += 1
        self.prihvaceno += n
        # posle odbijanja se ostatak baca neispitan, pa ne ulazi u imenilac
        self.provereno += n + 1 if n < self.gamma else self.gamma
        self.tokena += n + 1
        self.histogram_n[n] += 1
        self.max_odstupanje = max(self.max_odstupanje, odstupanje)

    def rezultat(self, c=None, t_baseline=None):
        a = alfa(self.prihvaceno, self.provereno)
        r = {
            "alfa": a,
            "iteracija": self.iteracija,
            "provereno": self.provereno,
            "tokena": self.tokena,
            "tokena_po_pozivu_target": self.tokena / self.iteracija,
            "teorijski_tokena_po_pozivu": ocekivano_tokena(a, self.gamma),
            "max_odstupanje_raspodele": self.max_odstupanje,
            "vreme_s": self.vreme,
        }
        if c is not None:
            r["c"] = c
            r["teorijsko_ubrzanje"] = teorijsko_ubrzanje(a, self.gamma, c)
        if t_baseline is not None:
            r["izmereno_ubrzanje"] = izmereno_ubrzanje(t_baseline, self.vreme)
        return r
