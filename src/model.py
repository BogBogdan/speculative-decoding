import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

teacher_path = "Qwen/Qwen2.5-14B"
student_path = "Qwen/Qwen2.5-0.5B"

_tokenizer = None
_teacher = None
_student = None


def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(student_path)
    return _tokenizer


def load_teacher():
    global _teacher
    if _teacher is None:
        _teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        _teacher.eval()
        for p in _teacher.parameters():
            p.requires_grad = False
    return _teacher


def load_student(dtype=torch.bfloat16):
    """dtype=torch.float32 za trening, bfloat16 za inferencu.

    Ucitavanje u bf16 pa .float() posle NE vraca preciznost - tezine su vec
    zaokruzene na 7 bita mantise. Za trening ucitaj direktno u fp32.
    """
    global _student
    if _student is None:
        _student = AutoModelForCausalLM.from_pretrained(
            student_path,
            dtype=dtype,
            device_map="cuda:0",
        )
    return _student
