import torch


def lk_divergence(p, q):
    return 0.5 * (p - q).abs().sum().item()


def beta(p, q):
    return torch.minimum(p, q).sum().item()


def token_beta(p, q, x):
    return min(1.0, (p[x] / q[x]).item())

def alpha_from_beta(betas):
    if torch.is_tensor(betas):
        return betas.float().mean().item()
    betas = list(betas)
    return sum(betas) / len(betas)


def alpha_from_divergence(divergences):
    divergences = list(divergences)
    return 1.0 - sum(divergences) / len(divergences)


def empirical_alpha(accepted, offered):
    return accepted / offered


def topk_alpha(student_logits, topk_logits, topk_indices, mask=None):
    """Acceptance rate alpha from teacher-forced logits, without sampling.

    beta at one position is sum_x min(p(x), q(x)); alpha is its mean over
    positions. The teacher p is only known on its top-k, so the tail
    min(p, q) <= p is dropped -- a small underestimate of alpha.

    student_logits [B, G, V]   topk_* [B, G, K]   mask [B, G]
    """
    p = torch.softmax(topk_logits.float(), dim=-1)
    # gather before the full-vocab normalizer so [B, G, V] is never materialized in fp32
    lse = torch.logsumexp(student_logits.float(), dim=-1, keepdim=True)
    q = (student_logits.gather(-1, topk_indices.long()).float() - lse).exp()

    betas = torch.minimum(p, q).sum(-1)
    if mask is not None:
        betas = betas[mask]
    return alpha_from_beta(betas)


def accepted_distribution(a, gamma):
    p = [a ** i * (1.0 - a) for i in range(gamma)]
    p.append(a ** gamma)
    return p


def expected_tokens(a, gamma):
    if a >= 1.0:
        return float(gamma + 1)
    return (1.0 - a ** (gamma + 1)) / (1.0 - a)


def expected_accepted(a, gamma):
    return expected_tokens(a, gamma) - 1.0


def cost_ratio(t_draft, t_target):
    return t_draft / t_target


def is_worthwhile(a, c):
    return a > c


def theoretical_speedup(a, gamma, c):
    return expected_tokens(a, gamma) / (gamma * c + 1.0)


def optimal_gamma(a, c, max_gamma=64):
    best = max(range(1, max_gamma + 1), key=lambda g: theoretical_speedup(a, g, c))
    return best, theoretical_speedup(a, best, c)


def operations_factor(a, gamma, c_ops):
    return (gamma * c_ops + gamma + 1.0) / expected_tokens(a, gamma)


def measured_speedup(t_baseline, t_speculative):
    return t_baseline / t_speculative


if __name__ == "__main__":
    import os
    import sys

    os.environ["USER"] = "aleksa"
    os.environ["HF_HOME"] = "/data"
    sys.path.append("/home/mls07/speculative-decoding/train")

    from dataset import DistillationDataset
    from metrics import izmeri_c_vllm
    from model import load_student

    c, t_draft, t_target = izmeri_c_vllm()

    draft = load_student()
    draft.eval()
    device = next(draft.parameters()).device

    data = DistillationDataset("/home/mls07/data")
    sample = data[:4]
    ids = sample["generated_ids"].to(device)
    topk_logits = sample["topk_logits"].to(device)
    topk_indices = sample["topk_indices"].to(device)
    mask = sample["loss_mask"].to(device)

    with torch.no_grad():
        logits = draft(ids[:, :-1], logits_to_keep=topk_logits.shape[1], use_cache=False).logits
    p = torch.softmax(topk_logits.float(), dim=-1)
    q = torch.softmax(logits.float(), dim=-1).gather(-1, topk_indices.long())
    a = alpha_from_beta(torch.minimum(p, q).sum(-1)[mask])

    print(f"alpha = {a:.3f}   c = {c:.3f}   ({t_draft * 1e3:.2f} ms / {t_target * 1e3:.2f} ms per step)")
    print(f"worthwhile (alpha > c): {is_worthwhile(a, c)}\n")
    print(f"{'gamma':>6}{'E[tokens]':>12}{'speedup':>10}")
    for g in range(1, 9):
        print(f"{g:>6}{expected_tokens(a, g):>12.2f}{theoretical_speedup(a, g, c):>10.2f}")

    g_opt, s_opt = optimal_gamma(a, c)
    print(f"\noptimal gamma = {g_opt}, speedup = {s_opt:.2f}x")
