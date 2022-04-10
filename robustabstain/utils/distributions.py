import torch

def rademacher(shape: torch.Size) -> torch.tensor:
    """Returns a rademacher distribution (https://en.wikipedia.org/wiki/Rademacher_distribution)
    tensor of the given shape.

    Args:
        shape (torch.Shape): Shape of the returned tensor.

    Returns:
        torch.tensor: Random rademacher tensor.
    """

    return (2 * torch.randint(low=0, high=2, size=shape) - 1)