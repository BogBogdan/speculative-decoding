import torch
import torch.nn.functional as F
from transformers import DynamicCache


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


def _sample(probs):
    return torch.multinomial(probs, 1).item()


def _residual_sample(p, q):
    residual = (p - q).clamp(min=0)
    residual = residual / residual.sum()
    return _sample(residual)


@torch.no_grad()
def speculative_generate(draft, target, input_ids, gamma=4, max_new_tokens=128, eos_token_id=None, merenja=None):
    device = input_ids.device
    committed = input_ids.clone()

    draft_cache = DynamicCache()
    target_cache = DynamicCache()

    draft_logits = draft(input_ids=input_ids, past_key_values=draft_cache, use_cache=True).logits[0, -1]

    # Target's forward is done once per round over [pending_token] + gamma new
    # candidates, so it stays one token behind: pending_token is real but not
    # yet run through the target model / added to target_cache.
    pending_token = input_ids[:, -1:]
    if input_ids.shape[1] > 1:
        target(input_ids=input_ids[:, :-1], past_key_values=target_cache, use_cache=True)

    n_generated = 0
    while n_generated < max_new_tokens:
        candidates = []
        qs = []
        cur_logits = draft_logits
        for _ in range(gamma):
            q_i = F.softmax(cur_logits.float(), dim=-1)
            x_i = _sample(q_i)
            candidates.append(x_i)
            qs.append(q_i)
            cur_logits = draft(
                input_ids=torch.tensor([[x_i]], device=device),
                past_key_values=draft_cache,
                use_cache=True,
            ).logits[0, -1]

        target_input = torch.cat([pending_token, torch.tensor([candidates], device=device)], dim=1)
        target_logits = target(input_ids=target_input, past_key_values=target_cache, use_cache=True).logits[0]

        n_accepted = gamma
        new_token = None
        for i in range(gamma):
            p_i = F.softmax(target_logits[i].float(), dim=-1)
            if speculative_sampling(p_i, qs[i], candidates[i]):
                continue
            n_accepted = i
            new_token = _residual_sample(p_i, qs[i])
            break

        if new_token is None:
            bonus_p = F.softmax(target_logits[gamma].float(), dim=-1)
            new_token = _sample(bonus_p)
        else:
            draft_cache.crop(gamma - n_accepted)
            target_cache.crop(gamma - n_accepted)

        new_tokens = candidates[:n_accepted] + [new_token]
        committed = torch.cat([committed, torch.tensor([new_tokens], device=device)], dim=1)
        n_generated += len(new_tokens)

        if merenja is not None:
            merenja.dodaj(n_accepted)

        if eos_token_id is not None and eos_token_id in new_tokens:
            break

        pending_token = torch.tensor([[new_token]], device=device)
        draft_logits = draft(input_ids=pending_token, past_key_values=draft_cache, use_cache=True).logits[0, -1]

    return committed[:, : input_ids.shape[1] + max_new_tokens]