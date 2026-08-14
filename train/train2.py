import os
import sys
os.environ["USER"] = "aleksa"
os.environ["HF_HOME"] = "/data"
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/mls07/speculative-decoding/src")
from model import load_tokenizer, student_path

from dataset import DistillationDataset

GEN_LEN = 256

DATA_DIR = "/home/mls07/data"
OUTPUT_DIR = "/home/mls07/data/draft-distilled2"

INFER_PROMPT = "Explain KV caching in one sentence."




def kd_loss(student_logits, topk_logits, topk_indices, mask):
    # student_logits [B, G, V]   topk_* [B, G, K]   mask [B, G]
    teacher_logp = F.log_softmax(topk_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1, dtype=torch.float32).gather(-1, topk_indices.long())
    ce = -(teacher_logp.exp() * student_logp).sum(-1)   # [B, G]
    m = mask.to(ce.dtype)
    return (ce * m).sum() / m.sum().clamp(min=1)


def entropy_floor(topk_logits, mask):
    logp = F.log_softmax(topk_logits.float(), dim=-1)
    H = -(logp.exp() * logp).sum(-1)
    m = mask.to(H.dtype)
    return ((H * m).sum() / m.sum().clamp(min=1)).item()


@torch.no_grad()
def evaluate(student, batches):
    student.eval()
    total = 0.0
    for batch in batches:
        input_ids = batch["generated_ids"][:, :-1]
        student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits
        total += kd_loss(student_logits, batch["topk_logits"],
                         batch["topk_indices"], batch["loss_mask"]).item()
    student.train()
    return total / len(batches)


@torch.no_grad()
def quick_inference(student, tokenizer, prompt, device, max_new_tokens=50):
    student.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = student.generate(
        **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
    )
    student.train()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def plot_losses(steps, losses, epoch_steps, epoch_losses, val_losses, floor, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, alpha=0.3, label="step loss")
    plt.plot(epoch_steps, epoch_losses, marker="o", label="train loss")
    plt.plot(epoch_steps, val_losses, marker="s", label="val loss")
    plt.axhline(floor, color="red", ls="--", label=f"entropy floor {floor:.3f}")
    plt.xlabel("optimizer step")
    plt.ylabel("KD loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_batches(rows, n_micro, batch_size, device):
    batches = []
    for b in range(n_micro):
        chunk = [rows[b * batch_size + j] for j in range(batch_size)]
        batches.append({k: torch.stack([c[k] for c in chunk]).to(device) for k in chunk[0]})
    return batches


def main():
    BATCH_SIZE = 4
    ACCUM = 8
    EPOCHS = 5

    N_TRAIN_ROWS = 640
    N_VAL_ROWS = 128
    N_TRAIN_MICRO = N_TRAIN_ROWS // BATCH_SIZE   # 160 -> 20 optimizer steps/epoch
    N_VAL_MICRO = N_VAL_ROWS // BATCH_SIZE       # 32

    tokenizer = load_tokenizer()
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        dtype=torch.float32,
        device_map="cuda:0",
    )
    student.train()
    device = next(student.parameters()).device

    dataset = DistillationDataset(DATA_DIR)
    if len(dataset) < N_TRAIN_ROWS + N_VAL_ROWS:
        raise ValueError(f"dataset has {len(dataset)} rows, need {N_TRAIN_ROWS + N_VAL_ROWS}")

    train_rows = Subset(dataset, range(N_TRAIN_ROWS))
    val_rows = Subset(dataset, range(N_TRAIN_ROWS, N_TRAIN_ROWS + N_VAL_ROWS))

    batches = make_batches(train_rows, N_TRAIN_MICRO, BATCH_SIZE, device)
    val_batches = make_batches(val_rows, N_VAL_MICRO, BATCH_SIZE, device)

    floor = entropy_floor(
        torch.cat([b["topk_logits"] for b in batches]),
        torch.cat([b["loss_mask"] for b in batches]),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "overfit_curve.png")
    steps, losses = [], []
    epoch_steps, epoch_losses, val_losses = [], [], []

    epoch_steps.append(0)
    epoch_losses.append(evaluate(student, batches))
    val_losses.append(evaluate(student, val_batches))
    print(f"epoch  -1 step    0 train {epoch_losses[0]:.4f} val {val_losses[0]:.4f}")
    print(f"  sample: {quick_inference(student, tokenizer, INFER_PROMPT, device)!r}")

    step = 0
    accum_loss = 0.0
    print(f"{len(batches)} train micro-batches, {len(val_batches)} val micro-batches")
    for epoch in range(EPOCHS):
        mean = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(batches):
            input_ids = batch["generated_ids"][:, :-1]
            student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits

            loss = kd_loss(student_logits, batch["topk_logits"],
                           batch["topk_indices"], batch["loss_mask"])
            mean += loss.item()
            (loss / ACCUM).backward()
            accum_loss += loss.item()

            if (i + 1) % ACCUM != 0:
                continue

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            steps.append(step)
            losses.append(accum_loss / ACCUM)
            accum_loss = 0.0

        mean /= len(batches)
        val = evaluate(student, val_batches)
        epoch_steps.append(step)
        epoch_losses.append(mean)
        val_losses.append(val)
        print(f"epoch {epoch:3d} step {step:4d} train {mean:.4f} val {val:.4f}")
        print(f"  sample: {quick_inference(student, tokenizer, INFER_PROMPT, device)!r}")
        plot_losses(steps, losses, epoch_steps, epoch_losses, val_losses, floor, plot_path)

    print(f"curve at {plot_path}")


if __name__ == "__main__":
    main()
