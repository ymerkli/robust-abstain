import re

def latex_norm(adv_norm: str, no_l: bool = False) -> str:
    """Convert adversarial norm string to latex math string.

    Args:
        adv_norm (str): Adversarial norm (e.g. 'Linf', 'L2', etc.).
        no_l (bool, optional): Whether to return L2 or just 2. Defaults to False.

    Returns:
        str: Latex math-ified string (e.g. 'L_{\infty}', 'L_{2}')
    """
    match = re.match(r'^L(\d+)$', adv_norm)
    if adv_norm == 'Linf':
        p = '\infty'
    elif match:
        p = match.group(1)
    else:
        raise ValueError(f'Error: invalid norm {adv_norm}')

    norm = p if no_l else f'L_{{{p}}}'
    return norm

