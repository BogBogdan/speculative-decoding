import glob
import os

import torch
from torch.utils.data import Dataset


class DistillationDataset(Dataset):
    def __init__(self, data_dir):
        paths = sorted(glob.glob(os.path.join(data_dir, "batch_*.pt")))
        if not paths:
            raise FileNotFoundError(f"no batch_*.pt files found in {data_dir}")

        generated_ids, topk_logits, topk_indices = [], [], []
        for path in paths:
            batch = torch.load(path, map_location="cpu")
            generated_ids.append(batch["generated_ids"])
            topk_logits.append(batch["topk_logits"])
            topk_indices.append(batch["topk_indices"])

        self.generated_ids = torch.cat(generated_ids, dim=0)
        self.topk_logits = torch.cat(topk_logits, dim=0)
        self.topk_indices = torch.cat(topk_indices, dim=0)

    def __len__(self):
        return self.generated_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "generated_ids": self.generated_ids[idx],
            "topk_logits": self.topk_logits[idx],
            "topk_indices": self.topk_indices[idx],
        }
