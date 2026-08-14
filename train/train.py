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

VAL_EVERY = 200      # koraka optimizatora izmedju dve validacije
VAL_BATCHEVA = 50    # koliko batcheva po validaciji, umesto celog val skupa
EPOHA = 10           # spusti ako val gubitak pocne da raste rano

def kd_loss(student_logits, topk_logits, topk_indices, maska=None):
    """Top-k destilacija: -suma_k p_ucitelj(k) * log q_student(k).

    Uciteljev recnik je siri od studentovog - Qwen2.5-14B ima vocab_size 152064,
    a 0.5B 151936. Indeksi preko studentove sirine se izbacuju iz uciteljeve
    raspodele; bez toga gather cita van granica tenzora, sto na CUDA-i ne puca
    nego vrati smece i pokvari gradijente.
    """
    V = student_logits.shape[-1]
    vazi = topk_indices < V
    idx = topk_indices.long().clamp(max=V - 1)          # da gather ostane u granicama

    # -inf na nevazecim mestima -> teacher_p tamo je 0, pa ne doprinose gubitku
    teacher_p = torch.softmax(topk_logits.float().masked_fill(~vazi, float("-inf")), dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1, dtype=torch.float32).gather(-1, idx)

    po_poziciji = -(teacher_p * student_logp).sum(-1)   # [B, GEN_LEN]
    if maska is None:
        return po_poziciji.mean()
    return (po_poziciji * maska).sum() / maska.sum().clamp(min=1)


def maska_validnih(generated_ids, eos_id, gen_len):
    """Kad generate zavrsi ranije na EOS-u, ostatak sekvence dopuni EOS tokenima.
    Ti ciljevi nisu pravi tekst - student bi naucio da uvek izbacuje EOS.
    Zadrzava se prvi EOS (njega treba nauciti), sve posle njega se izbacuje.
    """
    ciljevi = generated_ids[:, -gen_len:]
    je_eos = (ciljevi == eos_id).long()
    return (je_eos.cumsum(-1) <= 1).float()

@torch.no_grad()
def evaluate(student, loader, device, eos_id, max_batcheva=VAL_BATCHEVA):
    """Validacija na ogranicenom broju batcheva.

    Ako se prolazi kroz CEO val skup na svakih VAL_EVERY koraka, ukupna cena
    validacije raste sa N^2 dok trening raste sa N - pa na velikom skupu
    validacija pojede vise vremena nego sam trening.
    """
    student.eval()
    total_loss, total_batches = 0.0, 0
    for batch in loader:
        if total_batches >= max_batcheva:
            break
        generated_ids = batch["generated_ids"].to(device)
        topk_logits = batch["topk_logits"].to(device)
        topk_indices = batch["topk_indices"].to(device)

        input_ids = generated_ids[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits
        maska = maska_validnih(generated_ids, eos_id, GEN_LEN)
        total_loss += kd_loss(student_logits, topk_logits, topk_indices, maska).item()
        total_batches += 1

    student.train()
    return total_loss / max(total_batches, 1)

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
    eos_id = tokenizer.eos_token_id
    student = load_student(dtype=torch.float32)   # trening u fp32, ne bf16 pa .float()
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

    # 5e-4 je vrednost iz overfit_check.py, gde je namerno visoka da bi jedan batch
    # brzo pao. Za pun trening 0.5B modela to je 10x previse i vodi u kolaps.
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    step = 0
    running_loss = 0.0
    for epoch in range(EPOHA):
        # cisti gradijent na pocetku svake epohe: ako len(loader) nije deljivo sa 8,
        # zaostatak poslednje grupe bi se prelio u prvi korak sledece epohe
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            generated_ids = batch["generated_ids"].to(device)
            topk_logits = batch["topk_logits"].to(device)
            topk_indices = batch["topk_indices"].to(device)
            input_ids = generated_ids[:, :-1]

            # tezine ostaju fp32, mnozenja se rade u bf16 - oko 2x brze,
            # bez gubitka preciznosti koji donosi ucitavanje modela u bf16
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(input_ids, logits_to_keep=GEN_LEN,
                                         use_cache=False).logits

            maska = maska_validnih(generated_ids, eos_id, GEN_LEN)
            loss = kd_loss(student_logits, topk_logits, topk_indices, maska)
            (loss / 8).backward()
            # gubitak se sabira kao i gradijent, da kriva prikaze prosek grupe a ne
            # samo poslednji mikro-batch
            running_loss += loss.detach() / 8

            if (i + 1) % 8 != 0:
                continue

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            train_steps.append(step)
            train_losses.append(running_loss.item())
            running_loss = 0.0
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {train_losses[-1]:.4f}")

            if step % VAL_EVERY == 0:
                val_loss = evaluate(student, val_loader, device, eos_id)
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