import os
import random
import sys

os.environ.setdefault("USER", "aleksa")
os.environ["HF_HOME"] = "/data"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
# vLLM's V1 engine normally spawns EngineCore as a separate process and
# handshakes over ZMQ.  That deadlocks when the parent has no attached stdin --
# i.e. any nohup/detached launch, which is how a multi-hour run gets started.
# Running the engine in-process avoids the spawn entirely.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

sys.path.append("/home/mls07/speculative-decoding/src")
from model import load_tokenizer, student_path, teacher_path

SEQUENCE_LEN = 128
GEN_LEN = 256
TOP_K = 50

N_ROWS = 30000

START_OFFSET = 3000
ROWS_PER_FILE = 512
SEED = 0
GPU_MEM = 0.90

OUT_DIR = "/home/mls07/data"
PREFIX_TXT = "/data/train/prefix-train.txt"
PREFIX_CACHE = "/home/mls07/data/prefix_pool.pt"

STUDENT_VOCAB = 151936


def build_prefix_pool(tokenizer):
    if os.path.exists(PREFIX_CACHE):
        pool = torch.load(PREFIX_CACHE, map_location="cpu")
        print(f"prefix pool: {pool.shape[0]:,} cached windows from {PREFIX_CACHE}")
        return pool

    print(f"building prefix pool from {PREFIX_TXT} (one-time, a few minutes)")
    paragraphs = open(PREFIX_TXT, encoding="utf-8").read().split("\n\n")
    rng = random.Random(SEED)
    windows = []

    CHUNK = 5000
    for start in range(0, len(paragraphs), CHUNK):
        chunk = paragraphs[start:start + CHUNK]
        encoded = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            if len(ids) < SEQUENCE_LEN:
                continue
            offset = rng.randint(0, len(ids) - SEQUENCE_LEN)
            windows.append(ids[offset:offset + SEQUENCE_LEN])
        if (start // CHUNK) % 20 == 0:
            print(f"  {start + len(chunk):,}/{len(paragraphs):,} paragraphs, "
                  f"{len(windows):,} windows")

    pool = torch.tensor(windows, dtype=torch.int64)
    torch.save(pool, PREFIX_CACHE)
    print(f"prefix pool: {pool.shape[0]:,} windows -> {PREFIX_CACHE}")
    return pool


def topk_from_logprobs(step_logprobs):
    ranked = sorted(step_logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True)
    idx, val, dropped = [], [], 0
    for token_id, lp in ranked:
        if token_id >= STUDENT_VOCAB:
            dropped += 1
            continue
        idx.append(token_id)
        val.append(lp.logprob)
        if len(idx) == TOP_K:
            break
    return idx, val, dropped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = load_tokenizer()
    pool = build_prefix_pool(tokenizer)

    if N_ROWS > pool.shape[0]:
        raise SystemExit(
            f"asked for {N_ROWS:,} rows but the pool holds {pool.shape[0]:,} "
            f"distinct prefixes; lower N_ROWS or allow prefix reuse"
        )
        
    g = torch.Generator().manual_seed(SEED)
    chosen = pool[torch.randperm(pool.shape[0], generator=g)[:N_ROWS]]
    print(f"writing rows {START_OFFSET}..{START_OFFSET + N_ROWS - 1} into {OUT_DIR}")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=teacher_path,
        tokenizer=student_path,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM,
        max_model_len=SEQUENCE_LEN + GEN_LEN + 8,
        max_logprobs=TOP_K,
        enable_prefix_caching=False,
        seed=SEED,
    )
    params = SamplingParams(
        max_tokens=GEN_LEN,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        logprobs=TOP_K,
        ignore_eos=True,
    )

    total_dropped = 0
    for start in range(0, N_ROWS, ROWS_PER_FILE):
        path = os.path.join(OUT_DIR, f"batch_{START_OFFSET + start:06d}.pt")
        if os.path.exists(path):
            print(f"exists, skipping {path}")
            continue

        prefixes = chosen[start:start + ROWS_PER_FILE]
        prompts = [TokensPrompt(prompt_token_ids=p.tolist()) for p in prefixes]
        outputs = llm.generate(prompts, params, use_tqdm=False)

        gen_ids, all_idx, all_val = [], [], []
        for out in outputs:
            completion = out.outputs[0]
            if len(completion.token_ids) != GEN_LEN:
                raise RuntimeError(
                    f"expected {GEN_LEN} tokens, got {len(completion.token_ids)}; "
                    f"is ignore_eos set?"
                )
            row_idx, row_val = [], []
            for step_logprobs in completion.logprobs:
                idx, val, dropped = topk_from_logprobs(step_logprobs)
                total_dropped += dropped
                row_idx.append(idx)
                row_val.append(val)
            gen_ids.append(list(completion.token_ids))
            all_idx.append(row_idx)
            all_val.append(row_val)

        generated = torch.tensor(gen_ids, dtype=torch.int64)
        torch.save(
            {
                "prefix_ids": prefixes.to(torch.int64),
                "generated_ids": torch.cat([prefixes.to(torch.int64), generated], dim=1),
                "topk_logits": torch.tensor(all_val, dtype=torch.float16),
                "topk_indices": torch.tensor(all_idx, dtype=torch.int32),
            },
            path,
        )
        done = min(start + ROWS_PER_FILE, N_ROWS)
        print(f"[{done}/{N_ROWS}] saved {path}")

    if total_dropped:
        print(f"note: dropped {total_dropped:,} top-K entries above the student vocab")
    print(f"done -> {OUT_DIR}")
    print("check the new data with:  python entropy_floor.py")


if __name__ == "__main__":
    main()
