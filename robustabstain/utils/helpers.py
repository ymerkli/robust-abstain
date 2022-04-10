import re
import torch
import numpy as np
from fractions import Fraction
from typing import List, Union


def has_attr(obj: object, k: str) -> bool:
    """Checks both that obj.k exists and is not equal to None

    Args:
        obj (object): object to check for attribute.
        k (str): attribute to check for.

    Returns:
        bool: True if obj has attribute k.
    """
    try:
        return (getattr(obj, k) is not None)
    except KeyError as e:
        return False
    except AttributeError as e:
        return False


def convert_floatstr(eps: str) -> float:
    """Converts a string that is either a float or a fraction into a float.

    Args:
        eps (str): A string of a float (e.g. 0.1) or a fraction (e.g. 1/10, 1_10)

    Returns:
        float: eps as float.
    """
    if type(eps) in [float, int]:
        return eps

    match = re.match(r'^\d+(\/|_)\d+$', eps)
    if match:
        num, den = eps.split(match.group(1))
        eps = float(num) / float(den)
    else:
        try:
            eps = float(eps)
        except:
            raise ValueError(f'Error: string {eps} is neither a float nor a fraction.')

    return eps


def loggable_floatstr(eps: str) -> str:
    """Convert a string of a float into a logging friendly version.

    Args:
        eps (str): A fraction or float as string (e.g. '1/2')

    Returns:
        str: Logging friendly eps string.
    """
    if re.match(r'^\d+(\/|_)\d+$', eps):
        eps = eps.replace('/', '_')
    else:
        try:
            _ = float(eps)
        except:
            raise ValueError(f'Error: string {eps} is neither a float nor a fraction.')

    return eps


def pretty_floatstr(eps: str) -> str:
    """Convert a stringfied float into a pretty stringified float.

    Args:
        eps (str): A fraction or float as loggable string (e.g. '1_2')

    Returns:
        str: Pretty stringified float.
    """
    return eps.replace('_', '/')


def nice_floatstr(eps_list: Union[List[float], float], digit_limit: int = 5) -> List[str]:
    """Convert a float to string and in case the number of digits exceeds the digit_limit,
    finds an approximate fraction.

    Args:
        eps_list (Union[List[float], float]): The float(s) to convert.
        digit_limit (int): Limit on number of digits after decimal point for which
            a fraction is produced.

    Returns:
        List[str]: List of converted fractions represented as strings.
    """
    if type(eps_list) is not list:
        eps = [eps_list]

    eps_nice_list = []
    for eps in eps_list:
        eps_nice, num_digits = str(eps), 0
        match = re.match(r'\d+(\.\d+)?', eps_nice)
        if match:
            if match.group(1) is not None:
                num_digits = len(match.group(1))-1
        else:
            raise ValueError(f'Error: {eps} is not a valid float.')

        if num_digits > digit_limit:
            eps_nice = str(Fraction(eps).limit_denominator(max_denominator=1000)).replace('/', '_')

    return eps_nice


def normstr2normp(normstr: str) -> float:
    """Convert a norm string ('Linf', 'L2', etc.) to the
    exective number p of the Lp norm.

    Args:
        normstr (str): Norm str ('Linf', 'L2', etc.)

    Returns:
        float: Float p of given Lp norm.
    """
    match = re.match(r'^L(\S+)$', normstr)
    if match:
        p = match.group(1)
        try:
            return int(p)
        except ValueError:
            pass

        if match.group(1) == 'inf':
            return float('inf')

    raise ValueError(f'Error: unknown Lp norm {normstr}')


def check_indicator(indicator: Union[np.ndarray, torch.tensor]) -> bool:
    """Assert that indicator contains only 0s and 1s and convert type to int.

    Args:
        indicator (Union[np.ndarray, torch.tensor]): Indicator array.

    Returns:
        bool: Checked indicator array converted to int.
    """
    assert ((indicator == 1) | (indicator == 0)).all(), 'Error: indicator can to only contain 0s and 1s'
    indicator = indicator.int() if type(indicator) == torch.tensor else indicator.astype(int)

    return indicator


def multiply_eps(eps: str, n: int) -> str:
    """Multiply a string of a perturbation region by a constant factor.

    Args:
        eps (str): A fraction or float as loggable string (e.g. '1_2').
        n (int): Multiplicate factor. Must be positive.

    Returns:
        str: n*eps as loggable string.
    """
    assert n > 0, 'Error: n must be positive.'
    assert isinstance(n, int), 'Error: n must be an integer.'

    match = re.match(r'^(\d+)(\/|_)(\d+)$', eps)
    if match:
        # eps is a string of a fraction (e.g. 1/255)
        num = n * int(match.group(1))
        sep = match.group(2)
        denum = int(match.group(3))
        eps = f"{num}{sep}{denum}"
    else:
        try:
            eps = float(eps)
            eps *= n
            eps = str(eps)
        except:
            raise ValueError(f"Error: invalid float string {eps}")
    
    return eps
