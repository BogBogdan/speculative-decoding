import os
import sys
os.environ["USER"] = "aleksa"
os.environ["HF_HOME"] = "/data"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/mls07/speculative-decoding/src")
from model import load_student, load_tokenizer

from dataset import DistillationDataset

GEN_LEN = 256  

DATA_DIR = "/home/mls07/data"
OUTPUT_DIR = "/home/mls07/data/draft-distilled"

def kd_loss(student_logits, topk_logits, topk_indices):
    teacher_logp = F.log_softmax(topk_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1, dtype=torch.float32).gather(-1, topk_indices.long())
    return -(teacher_logp.exp() * student_logp).sum(-1).mean()

@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    total_loss, total_batches = 0.0, 0
    for batch in loader:
        generated_ids = batch["generated_ids"].to(device)
        topk_logits = batch["topk_logits"].to(device)
        topk_indices = batch["topk_indices"].to(device)

        input_ids = generated_ids[:, :-1]
        student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits
        total_loss += kd_loss(student_logits, topk_logits, topk_indices).item()
        total_batches += 1

    student.train()
    return total_loss / total_batches

def plot_losses(train_steps, train_losses, val_steps, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="train loss")
    plt.plot(val_steps, val_losses, label="val loss", marker="o")
    plt.xlabel("optimizer step")
    plt.ylabel("KD loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    tokenizer = load_tokenizer()
    student = load_student().float()
    student.train()
    device = next(student.parameters()).device

    dataset = DistillationDataset(DATA_DIR)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
    )

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.Adam(student.parameters(), lr=5e-4)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    step = 0
    optimizer.zero_grad()
    for epoch in range(10):
        for i, batch in enumerate(loader):
            generated_ids = batch["generated_ids"].to(device)
            topk_logits = batch["topk_logits"].to(device)
            topk_indices = batch["topk_indices"].to(device)
            input_ids = generated_ids[:, :-1]
            student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits

            loss = kd_loss(student_logits, topk_logits, topk_indices)
            (loss / 8).backward()

            if (i + 1) % 8 != 0:
                continue

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            train_steps.append(step)
            train_losses.append(loss.item())
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if step % 10 == 0:
                val_loss = evaluate(student, val_loader, device)
                val_steps.append(step)
                val_losses.append(val_loss)
                print(f"epoch {epoch} step {step} val_loss {val_loss:.4f}")
                plot_losses(train_steps, train_losses, val_steps, val_losses, plot_path)

        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch-{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        student.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"saved checkpoint to {epoch_dir}")
        continue

    student.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"saved final checkpoint to {OUTPUT_DIR}")
    print(f"loss curve at {plot_path}")


if __name__ == "__main__":
    main()