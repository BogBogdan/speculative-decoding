import os
import sys
os.environ["USER"] = "aleksa"
os.environ["HF_HOME"] = "/data"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/mls07/speculative-decoding/src")
from model import load_student, load_tokenizer

from dataset import DistillationDataset

GEN_LEN = int(os.environ.get("GEN_LEN", 256))   # mora se poklopiti sa generatorom podataka

DATA_DIR = os.environ.get("DATA_DIR", "/home/mls07/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/mls07/data/draft-distilled")

MICRO_BATCH = int(os.environ.get("MICRO_BATCH", 8))
ACCUM = int(os.environ.get("ACCUM", 8))         # efektivni batch = MICRO_BATCH * ACCUM
EPOHA = int(os.environ.get("EPOHA", 10))
LR = float(os.environ.get("LR", 5e-5))
VAL_UDEO = float(os.environ.get("VAL_UDEO", 0.1))
EVAL_BATCH = 16
EVAL_SVAKIH = int(os.environ.get("EVAL_SVAKIH", 50))  # koraka izmedju tacaka na val krivoj
BRZI_VAL = 512             # koliko uzoraka ide u periodicni eval; pun val tek na kraju epohe

def kd_po_poziciji(student_logits, topk_logits, topk_indices):
    """Top-k destilacija: -suma_k p_ucitelj(k) * log q_student(k), po poziciji.

    Uciteljev recnik je siri od studentovog - Qwen2.5-14B ima vocab_size 152064,
    a 0.5B 151936. Indeksi preko studentove sirine se izbacuju iz uciteljeve
    raspodele; bez toga gather cita van granica tenzora, sto na CUDA-i ne puca
    nego vrati smece i pokvari gradijente.

    Racuna se u fp32 i pod autocast-om: .float() i dtype=torch.float32 vracaju
    ulaze iz bf16, pa mesovita preciznost ne dira numeriku samog gubitka.
    """
    V = student_logits.shape[-1]
    vazi = topk_indices < V
    idx = topk_indices.long().clamp(max=V - 1)          # da gather ostane u granicama

    # -inf na nevazecim mestima -> teacher_p tamo je 0, pa ne doprinose gubitku
    teacher_p = torch.softmax(topk_logits.float().masked_fill(~vazi, float("-inf")), dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1, dtype=torch.float32).gather(-1, idx)

    return -(teacher_p * student_logp).sum(-1)          # [B, GEN_LEN]


def kd_loss(student_logits, topk_logits, topk_indices, maska=None):
    po_poziciji = kd_po_poziciji(student_logits, topk_logits, topk_indices)
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
def evaluate(student, loader, device, eos_id):
    """Zbir gubitka i broj vazecih pozicija se akumuliraju odvojeno.

    Prosek preko batcheva bi poslednjem, nepunom batchu dao istu tezinu kao
    punom; isto vazi i za sekvence koje su ranije stale na EOS-u pa imaju manje
    vazecih pozicija od ostalih.
    """
    student.eval()
    zbir = torch.zeros((), device=device, dtype=torch.float32)
    broj = torch.zeros((), device=device, dtype=torch.float32)
    for batch in loader:
        generated_ids = batch["generated_ids"].to(device, non_blocking=True)
        topk_logits = batch["topk_logits"].to(device, non_blocking=True)
        topk_indices = batch["topk_indices"].to(device, non_blocking=True)

        input_ids = generated_ids[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits
        maska = maska_validnih(generated_ids, eos_id, GEN_LEN)
        po_poziciji = kd_po_poziciji(student_logits, topk_logits, topk_indices)
        # jedan .item() na kraju umesto po batchu - inace svaki poziv sinhronizuje GPU
        zbir += (po_poziciji * maska).sum()
        broj += maska.sum()

    student.train()
    return (zbir / broj.clamp(min=1)).item()

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
    # A100 default-uje na pune fp32 matmule (19.5 TFLOPS) umesto TF32 tensor
    # core-ova (156 TFLOPS). TF32 zadrzava fp32 eksponent i akumulira u fp32,
    # samo skracuje mantisu ulaza na 10 bita.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = load_tokenizer()
    eos_id = tokenizer.eos_token_id
    # Tezine ostaju fp32, racun ide u bf16 preko autocast-a (mesovita preciznost).
    # Cist bf16 trening ovde ne radi: Adam update je reda lr=5e-5, tipicna tezina
    # ~0.02, dakle relativna promena 2.5e-3 - manja od bf16 ulp-a (2^-8 = 3.9e-3),
    # pa bi se vecina koraka zaokruzila na nulu.
    student = load_student(dtype=torch.float32)
    student.train()
    device = next(student.parameters()).device

    dataset = DistillationDataset(DATA_DIR)
    val_size = int(len(dataset) * VAL_UDEO)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
    )

    loader = DataLoader(train_dataset, batch_size=MICRO_BATCH, shuffle=True)

    # Periodicni eval ide na fiksnom podskupu. Podskup mora biti isti pri svakom
    # pozivu - da se val kriva menja zbog modela, a ne zbog toga sto je izvucen
    # drugi uzorak. Pun val set se meri jednom po epohi.
    brzi_val = Subset(val_dataset, range(min(BRZI_VAL, len(val_dataset))))
    brzi_val_loader = DataLoader(brzi_val, batch_size=EVAL_BATCH, shuffle=False)
    pun_val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH, shuffle=False)

    # 5e-4 je vrednost iz overfit_check.py, gde je namerno visoka da bi jedan batch
    # brzo pao. Za pun trening 0.5B modela to je 10x previse i vodi u kolaps.
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    step = 0
    running_loss = 0.0
    for epoch in range(EPOHA):
        # cisti gradijent na pocetku svake epohe: ako len(loader) nije deljivo sa
        # ACCUM, zaostatak poslednje grupe bi se prelio u prvi korak sledece epohe
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            generated_ids = batch["generated_ids"].to(device, non_blocking=True)
            topk_logits = batch["topk_logits"].to(device, non_blocking=True)
            topk_indices = batch["topk_indices"].to(device, non_blocking=True)
            input_ids = generated_ids[:, :-1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(input_ids, logits_to_keep=GEN_LEN, use_cache=False).logits

            maska = maska_validnih(generated_ids, eos_id, GEN_LEN)
            loss = kd_loss(student_logits, topk_logits, topk_indices, maska)
            (loss / ACCUM).backward()
            # gubitak se sabira kao i gradijent, da kriva prikaze prosek grupe a ne
            # samo poslednji mikro-batch
            running_loss += loss.detach() / ACCUM

            if (i + 1) % ACCUM != 0:
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

            if step % EVAL_SVAKIH == 0:
                val_loss = evaluate(student, brzi_val_loader, device, eos_id)
                val_steps.append(step)
                val_losses.append(val_loss)
                print(f"epoch {epoch} step {step} val_loss {val_loss:.4f}")
                # crtanje je matplotlib figura + savefig, reda 200ms; na svakom
                # koraku bi se nakupilo u desetine minuta po celom treningu
                plot_losses(train_steps, train_losses, val_steps, val_losses, plot_path)

        pun_val = evaluate(student, pun_val_loader, device, eos_id)
        print(f"epoch {epoch} pun val_loss {pun_val:.4f}")
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