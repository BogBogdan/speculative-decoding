from get_dataset import get_prefix, get_random_prefix, tokenizuj_prefikse
import torch

if __name__ == "__main__":
    svi = get_prefix()
    uzorak = get_random_prefix(svi)
    ids = tokenizuj_prefikse(uzorak)

    print(f"ukupno: {len(svi)}")
    print(f"uzorak: {len(uzorak)}")
    print(f"tokenizovano: {len(ids)}")

    torch.save(ids, "data/prefiksi.pt")