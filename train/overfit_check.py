import os
import sys

os.environ["USER"] = "aleksa"
os.environ["HF_HOME"] = "/data"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("/home/mls07/speculative-decoding/src")
from model import load_student, load_tokenizer

from dataset import DistillationDataset

GEN_LEN = 256
DATA_DIR = "/home/mls07/data"
LR = 5e-4
STEPS = 200


# ista funkcija kao u treningu, da provera odrazava stvarne uslove
from train import kd_loss, maska_validnih


def main():
    student = load_student(dtype=torch.float32)   # isti uslovi kao u train.py
    student.train()
    device = next(student.parameters()).device

    dataset = DistillationDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    generated_ids = batch["generated_ids"].to(device)
    topk_logits = batch["topk_logits"].to(device)
    topk_indices = batch["topk_indices"].to(device)
    input_ids = generated_ids[:, :-1]

    # maska mora i ovde: EOS dopuna ima skoro nultu entropiju, pa bi pod
    # ispao lazno nizak i provera bi bila neupotrebljiva
    eos_id = load_tokenizer().eos_token_id
    maska = maska_validnih(generated_ids, eos_id, GEN_LEN)
    print(f"vazecih pozicija: {int(maska.sum())} / {maska.numel()}"
          f"  ({100 * maska.mean().item():.1f}%)")

    teacher_logp = F.log_softmax(topk_logits.float(), dim=-1)
    po_poziciji = -(teacher_logp.exp() * teacher_logp).sum(-1)
    entropy_floor = (po_poziciji * maska).sum().item() / maska.sum().clamp(min=1).item()
    print(f"entropy floor for this batch: {entropy_floor:.4f}")
    print("   gubitak treba da se zaustavi TU, ne da padne ka nuli\n")

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    for step in range(STEPS):
        student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits
        loss = kd_loss(student_logits, topk_logits, topk_indices, maska)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == STEPS - 1:
            print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
