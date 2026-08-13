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


def load_student():
    global _student
    if _student is None:
        _student = AutoModelForCausalLM.from_pretrained(
            student_path,
            dtype=torch.bfloat16, # zameniti sa float32 ako se trenira, bfloat16 je za inference
            device_map="cuda:0",
        )
    return _student
