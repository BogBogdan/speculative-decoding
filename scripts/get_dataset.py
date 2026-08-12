from datasets import load_dataset
import random
import re
from transformers import AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B") 

def get_random_prefix(prefix, length=100, seed=42):
    if len(prefix) < length:
        return prefix
    else:
        rng = random.Random(seed)
        return rng.sample(prefix, length)

    
def get_dataset():
    return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

def process_row(row_text):
    if row_text.startswith(" = ") or len(row_text) < 300:
        return None

    t = row_text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")

    t = re.sub(r' ([,.;:!?%])', r'\1', t)
    t = re.sub(r'\( ', '(', t)
    t = re.sub(r' \)', ')', t)
    t = re.sub(r'\$ ', '$', t)

    return t

def get_prefix():
    prefix = []
    ds_test = get_dataset()
    for row in ds_test:

        processed_text = process_row(row['text'])
        if processed_text is not None:
            prefix.append(processed_text)

    # with open("prefix.txt", "w", encoding="utf-8") as f:
    #     for item in prefix:
    #         f.write("%s\n" % item)
    # print(prefix)
    return prefix

def tokenizuj_prefikse(prefixes, duzina=50):
    out = []
    for t in prefixes:
        ids = tok(t, return_tensors="pt").input_ids[:, :duzina]
        if ids.shape[-1] == duzina:
            out.append(ids)
    return out

if __name__ == "__main__":
    svi = get_prefix()
    uzorak = get_random_prefix(svi)
    ids = tokenizuj_prefikse(uzorak)

    print(f"ukupno: {len(svi)}")
    print(f"uzorak: {len(uzorak)}")
    print(f"tokenizovano: {len(ids)}")

    torch.save(ids, "../data/prefiksi.pt")