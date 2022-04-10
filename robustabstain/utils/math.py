import math
from typing import Union


def roundupto(x: Union[float, int], to: int) -> int:
    """Round up to next multiple of to.

    Args:
        x (Union[float, int]): Number to round up.
        to (int): Multiple to round up to.

    Returns:
        int: Rounded number.
    """
    return int(math.ceil(x / float(to))) * to


def rounddownto(x: Union[float, int], to: int) -> int:
    """Round down to next multiple of to.

    Args:
        x (Union[float, int]): Number to round down.
        to (int): Multiple to round down to.

    Returns:
        int: Rounded number.
    """
    return int(math.floor(x / float(to))) * to