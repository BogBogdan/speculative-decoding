"""Entropy floor of the KD loss.

CE(p_t, q_s) = H(p_t) + KL(p_t || q_s).  H(p_t) does not depend on the student,
so it is a constant offset: the kd_loss curve in train.py can never go below it,
and (loss - H) is the part that is actually being minimized.

Since kd_loss compares against log_softmax(topk_logits) -- i.e. the teacher
renormalized over its top-K -- the relevant floor is H of that renormalized
distribution, not the teacher's full-vocab entropy.

Reads only the recorded top-K tensors; no model is loaded. CPU is fine.
"""

import math
import sys

import torch
import torch.nn.functional as F

from dataset import DistillationDataset

DATA_DIR = "/home/mls07/data"


def main(data_dir=DATA_DIR):
    ds = DistillationDataset(data_dir)
    logits = ds.topk_logits.float()          # [N, G, K] raw teacher logits, top-K only
    mask = ds.loss_mask                      # [N, G]
    K = logits.shape[-1]

    logp = F.log_softmax(logits, dim=-1)     # exactly what kd_loss uses as the target
    p = logp.exp()
    H = -(p * logp).sum(-1)                  # [N, G] nats per position

    m = mask.float()
    kept = m.sum()
    floor = ((H * m).sum() / kept).item()

    print(f"rows {len(ds)}  gen_len {ds.gen_len}  K {K}")
    print(f"positions kept by loss_mask: {int(kept)} / {mask.numel()} "
          f"({100 * kept / mask.numel():.1f}%)")
    print()
    print(f"H(p_t) masked   (floor for train.py kd_loss) : {floor:.4f} nats "
          f"= {floor / math.log(2):.4f} bits")
    print(f"H(p_t) unmasked (floor for train2.py/overfit_check): "
          f"{H.mean().item():.4f} nats")
    print()

    Hm = H[mask]
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    vals = torch.quantile(Hm, torch.tensor(qs))
    print("per-token H quantiles (masked):")
    for q, v in zip(qs, vals.tolist()):
        print(f"  p{int(q * 100):<3} {v:.4f}")
    print(f"  frac with H < 0.01 (near-deterministic): {(Hm < 0.01).float().mean():.4f}")

    top1 = p.max(-1).values[mask]
    print(f"\nteacher top-1 prob (within top-{K}): mean {top1.mean():.4f}  "
          f"median {top1.median():.4f}")

    print("\nhow much of the floor comes from the tail of the top-K:")
    for j in [1, 2, 4, 8, 16, 32, K]:
        if j > K:
            continue
        lj = F.log_softmax(logits[..., :j], dim=-1)
        hj = -(lj.exp() * lj).sum(-1)
        print(f"  H over top-{j:>3} renormalized: {((hj * m).sum() / kept).item():.4f} nats")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
