import torch


def speculative_sampling(p, q, x, useRandom=True):
    if q[x] <= p[x]:
        return True
    elif useRandom is False:
        return False
    else:
        r = torch.rand(1).item()
        if r < p[x] / q[x]:
            return True
        else:
            return False