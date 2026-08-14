import gc
import os
import time

os.environ["USER"] = "aleksa"
os.environ["HF_HOME"] = "/data"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import torch
from vllm import LLM, SamplingParams

from model import student_path, teacher_path

PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
SHORT, LONG = 1, 129
REPEATS = 3

def generation_time(llm, max_tokens):
    params = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
    llm.generate([PROMPT], params, use_tqdm=False)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        llm.generate([PROMPT], params, use_tqdm=False)
        times.append(time.perf_counter() - t0)
    return sorted(times)[REPEATS // 2]

def decode_latency(model_path, gpu_memory_utilization):
    llm = LLM(
        model=model_path,
        tokenizer=student_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )
    latency = (generation_time(llm, LONG) - generation_time(llm, SHORT)) / (LONG - SHORT)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return latency

def student_latency():
    return decode_latency(student_path, 0.35)

def teacher_latency():
    return decode_latency(teacher_path, 0.9)

if __name__ == "__main__":
    t_draft = student_latency()
    t_target = teacher_latency()
    print(f"draft  {t_draft * 1e3:.2f} ms per step")
    print(f"target {t_target * 1e3:.2f} ms per step")
    print(f"c = {t_draft / t_target:.4f}")
