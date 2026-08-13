import time

import torch

from metrics import _sync, autoregresivno, perplexity
from notebooks.model import student, tokenizer, teacher 

EVAL_FILE = "/data/evaluation/prefix-evaluation.txt"
N_SEQUENCES = 20
MAX_LEN = 256
PROMPT_LEN = 32
GEN_TOKENS = 50
REPEATS = 3


def load_eval_sequences(path, n, max_len):
    with open(path, "r", encoding="utf-8") as f:
        paragraphs = [p for p in f.read().split("\n\n") if p.strip()]

    sequences = []
    for p in paragraphs[:n]:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_len]
        if ids.shape[1] >= 2:
            sequences.append(ids)
    return sequences


@torch.no_grad()
def top1_accuracy(model, sequences):
    device = next(model.parameters()).device
    correct, total = 0, 0
    for ids in sequences:
        ids = ids.to(device)
        preds = model(ids).logits[0, :-1].argmax(dim=-1)
        targets = ids[0, 1:]
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / total


@torch.no_grad()
def measure_inference_time(model, prompt_ids, gen_tokens, repeats):
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)

    autoregresivno(model, ids, gen_tokens)  # warmup
    _sync(device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        autoregresivno(model, ids, gen_tokens)
        _sync(device)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return avg, gen_tokens / avg


def benchmark(name, model, sequences, prompt_ids, gen_tokens, repeats):
    ppl = perplexity(model, sequences)
    acc = top1_accuracy(model, sequences)
    seconds, tokens_per_s = measure_inference_time(model, prompt_ids, gen_tokens, repeats)
    return {
        "model": name,
        "perplexity": ppl,
        "top1_accuracy": acc,
        "s_per_generation": seconds,
        "tokens_per_s": tokens_per_s,
    }


def print_results(results):
    header = f"{'model':<10}{'perplexity':>12}{'top1_acc':>10}{'tokens/s':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<10}{r['perplexity']:>12.3f}{r['top1_accuracy']:>10.3f}{r['tokens_per_s']:>10.2f}")


def main():
    sequences = load_eval_sequences(EVAL_FILE, N_SEQUENCES, MAX_LEN)
    prompt_ids = sequences[0][:, :PROMPT_LEN]

    results = []
    for name, model in [("teacher", teacher), ("student", student)]:
        model.eval()
        results.append(benchmark(name, model, sequences, prompt_ids, GEN_TOKENS, REPEATS))

    print_results(results)


if __name__ == "__main__":
    main()
