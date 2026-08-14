"""Where the KD loss starts, and what the starting value is made of.

CE = H(p_t) + KL(p_t || q~_s) - log M_s
     floor    shape error       leakage (student mass outside the teacher's top-K)

Run a student checkpoint over the recorded batches and report all three terms,
so a loss value can be read as "how far above the floor, and why".

    python initial_loss.py [checkpoint ...]

Defaults to the untrained base student, i.e. the loss at step 0.
"""

import math
import os
import sys

os.environ.setdefault("USER", "aleksa")
os.environ.setdefault("HF_HOME", "/data")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from dataset import DistillationDataset

GEN_LEN = 256
DATA_DIR = "/home/mls07/data"
N_ROWS = 256          # subset; the floor is stable to ~0.01 nats well below this
BATCH = 8


@torch.no_grad()
def measure(path, ds, n_rows=N_ROWS, batch=BATCH):
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, device_map="cuda:0"
    ).eval()
    dev = next(model.parameters()).device
    V = model.config.vocab_size

    ce_s = h_s = leak_s = n = 0.0
    for i in range(0, min(n_rows, len(ds)), batch):
        ids = ds.generated_ids[i:i + batch].to(dev)
        m = ds.loss_mask[i:i + batch].to(dev).float()
        tl = ds.topk_logits[i:i + batch].float().to(dev)
        ti = ds.topk_indices[i:i + batch].long().to(dev)

        logits = model(ids[:, :-1], logits_to_keep=GEN_LEN, use_cache=False).logits
        slp = F.log_softmax(logits, dim=-1, dtype=torch.float32).gather(-1, ti)
        tlp = F.log_softmax(tl, dim=-1)
        tp = tlp.exp()

        ce = -(tp * slp).sum(-1)                  # exactly kd_loss, per position
        H = -(tp * tlp).sum(-1)
        leak = -slp.exp().sum(-1).clamp(max=1.0).log()   # -log M_s

        ce_s += (ce * m).sum().item()
        h_s += (H * m).sum().item()
        leak_s += (leak * m).sum().item()
        n += m.sum().item()

    del model
    torch.cuda.empty_cache()
    return ce_s / n, h_s / n, leak_s / n, V, int(n)


def main(paths):
    ds = DistillationDataset(DATA_DIR)
    for path in paths:
        ce, H, leak, V, n = measure(path, ds)
        shape = ce - H - leak
        print(f"\n{path}   ({n} masked positions, vocab {V})")
        print(f"  CE  (= kd_loss)          {ce:8.4f} nats  = {ce / math.log(2):.4f} bits")
        print(f"    H(p_t)      floor      {H:8.4f}   ({100 * H / ce:.1f}% of the loss)")
        print(f"    KL shape    trainable  {shape:8.4f}   ({100 * shape / ce:.1f}%)")
        print(f"    -log M_s    leakage    {leak:8.4f}   ({100 * leak / ce:.1f}%)")
        print(f"  student mass on teacher top-K: {math.exp(-leak):.4f}")
        print(f"  for reference, uniform over vocab would give CE = {math.log(V):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Qwen/Qwen2.5-0.5B"])
