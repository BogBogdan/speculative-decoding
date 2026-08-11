from datasets import load_dataset
import random
import re

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

if __name__ == "__main__":
    prefix = get_prefix()
    print(f"{len(prefix)}")