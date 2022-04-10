import torch


def abstain_loss(logits: torch.tensor, targets: torch.tensor, variant: str = 'sum') -> torch.tensor:
    """Abstain loss as in Laidlaw et. al. (https://arxiv.org/abs/1911.11253)

    Args:
        logits (torch.tensor): Class logits. logits[:,-1] must be the abstain logits
        targets (torch.tensor): ground truth labels
        variant (str, optional): Abstain loss version ('sum', 'or')
            (l(1), l(2) in the paper by Laidlaw et. al.). Defaults to 'sum'.
    """
    abstain_logits = logits[:, -1]
    correct_logits = logits.gather(1, targets[:, None])[:, -1]

    loss = None
    if variant == 'sum':
        loss = - (
            torch.stack([
                abstain_logits,
                correct_logits
            ], dim=1).logsumexp(1) - logits.logsumexp(1)
        )
    elif variant == 'or':
        loss = (
            (logits.logsumexp(1) - correct_logits.unsqueeze(1).logsumexp(1))
            * (logits.logsumexp(1) - abstain_logits.unsqueeze(1).logsumexp(1))
        )
    else:
        raise ValueError(f'Error: unknown abstain loss variant {variant}')

    return loss
