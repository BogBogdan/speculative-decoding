"""Speculative decoding written by hand on top of two vLLM engines.

Each round: run the draft model GAMMA times (one token per call), then a single
target call that both verifies all GAMMA candidates and produces the bonus token.
The target verification is one forward pass because vLLM's `prompt_logprobs`
scores every prompt position at once -- position len(ctx)+i carries the target
distribution conditioned on ctx + candidates[:i], which is exactly p_i.
"""

import gc
import math
import os
import random
import time

os.environ["USER"] = "aleksa"
os.environ["HF_HOME"] = "/data"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from model import load_tokenizer, student_path, teacher_path

PROMPT = "The capital of France is"
GAMMA = 3
ROUNDS = 10
TOP_K = 20  # vLLM's default max_logprobs; the support we use for residual sampling
DRAFT_GPU = 0.08
TARGET_GPU = 0.84
SEED = 0


def probs(logprob_dict):
    return {tid: math.exp(lp.logprob) for tid, lp in logprob_dict.items()}


def residual_sample(p, q):
    # Exact residual sampling needs the full vocab; we only hold the target's
    # top-K, so the tail (total mass < K * min(p)) is dropped. Tokens outside the
    # draft's top-K get q = 0, which only overestimates the residual slightly.
    tokens = list(p)
    weights = [max(p[t] - q.get(t, 0.0), 0.0) for t in tokens]
    total = sum(weights)
    if total <= 0:
        return max(p, key=p.get)
    return random.choices(tokens, weights=weights, k=1)[0]


def draft_step(draft, token_ids, params):
    out = draft.generate([TokensPrompt(prompt_token_ids=token_ids)], params, use_tqdm=False)[0].outputs[0]
    return out.token_ids[0], probs(out.logprobs[0])


def target_verify(target, token_ids, params):
    """One target forward over ctx + candidates -> p_0..p_gamma and the bonus token."""
    out = target.generate([TokensPrompt(prompt_token_ids=token_ids)], params, use_tqdm=False)[0]
    # prompt_logprobs[t] is the distribution over position t given everything before it,
    # so the last GAMMA entries score the candidates. Entry 0 is always None (no context).
    ps = [probs(lp) for lp in out.prompt_logprobs[-GAMMA:]]
    bonus_id = out.outputs[0].token_ids[0]
    ps.append(probs(out.outputs[0].logprobs[0]))  # p_gamma, the distribution the bonus came from
    return ps, bonus_id


def speculative_round(draft, target, token_ids, draft_params, target_params):
    candidates, qs = [], []
    for _ in range(GAMMA):
        token_id, q = draft_step(draft, token_ids + candidates, draft_params)
        candidates.append(token_id)
        qs.append(q)

    ps, bonus_id = target_verify(target, token_ids + candidates, target_params)

    n_accepted = GAMMA
    new_token = bonus_id
    for i in range(GAMMA):
        p_x = ps[i].get(candidates[i], 0.0)
        q_x = qs[i][candidates[i]]
        if p_x >= q_x or random.random() < p_x / q_x:
            continue
        n_accepted = i
        new_token = residual_sample(ps[i], qs[i])
        break

    return candidates[:n_accepted] + [new_token], n_accepted


def main():
    random.seed(SEED)
    tokenizer = load_tokenizer()

    draft = LLM(
        model=student_path,
        dtype="bfloat16",
        gpu_memory_utilization=DRAFT_GPU,
        max_model_len=2048,
        seed=SEED,
    )
    target = LLM(
        model=teacher_path,
        tokenizer=student_path,
        dtype="bfloat16",
        gpu_memory_utilization=TARGET_GPU,
        max_model_len=2048,
        seed=SEED,
    )

    draft_params = SamplingParams(max_tokens=1, temperature=1.0, logprobs=TOP_K, ignore_eos=True)
    target_params = SamplingParams(
        max_tokens=1, temperature=1.0, logprobs=TOP_K, prompt_logprobs=TOP_K, ignore_eos=True
    )

    token_ids = tokenizer(PROMPT).input_ids
    n_prompt = len(token_ids)
    accepted_total = 0

    t0 = time.perf_counter()
    for r in range(ROUNDS):
        new_tokens, n_accepted = speculative_round(draft, target, token_ids, draft_params, target_params)
        token_ids = token_ids + new_tokens
        accepted_total += n_accepted
        print(
            f"round {r + 1:2d}  accepted {n_accepted}/{GAMMA}  "
            f"+{len(new_tokens)} tokens  {tokenizer.decode(new_tokens)!r}"
        )
    seconds = time.perf_counter() - t0

    n_generated = len(token_ids) - n_prompt
    print(f"\n{tokenizer.decode(token_ids)}\n")
    print(f"{n_generated} tokens in {ROUNDS} rounds, {seconds:.2f} s ({n_generated / seconds:.1f} tokens/s)")
    print(f"acceptance rate {accepted_total / (ROUNDS * GAMMA):.2f}")

    del draft, target
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
