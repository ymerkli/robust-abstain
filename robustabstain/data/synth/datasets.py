import numpy as np
import math
from typing import List, Tuple

from robustabstain.data.synth.generate import (
    gen_gauss, gen_spiral, gen_polynomial,
    gen_circle, gen_ellipse, gen_yline
)


SYNTH_DATASETS = [
    'twogauss', 'threegauss', 'threegausso', 'spiral', 'polynomial',
    'twocircle', 'threeyline', 'circletwogauss', 'ellipsecircle'
]


def two_gauss(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from two different Gaussian distributions.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Gaussian noise standard deviation. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    points0, labels0 = gen_gauss([0.25, 0.25], 0, math.ceil(n_samples/2), [noise, noise])
    points1, labels1 = gen_gauss([0.75, 0.75], 1, math.ceil(n_samples/2), [noise, noise])
    points = points0 + points1
    labels = labels0 + labels1

    return np.array(points, dtype=np.float32), labels


def three_gauss(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from three different Gaussian distributions.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Gaussian noise standard deviation. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    points01, labels01 = gen_gauss([0.25, 0.25], 0, math.ceil(31/64*n_samples), [noise, noise])
    points02, labels02 = gen_gauss([0.55, 0.8], 2, math.ceil(1/64*n_samples), [noise/3, noise/2])
    points1, labels1 = gen_gauss([0.75, 0.75], 1, math.ceil(n_samples/2), [noise, noise])
    points = points01 + points02 + points1
    labels = labels01 + labels02 + labels1

    return np.array(points, dtype=np.float32), labels


def three_gauss_o(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from three different Gaussian distributions: two 'blob' Gaussians and one
    oblong Gaussian.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Gaussian noise standard deviation. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    points01, labels01 = gen_gauss([0.25, 0.25], 0, math.ceil(15/32*n_samples), [noise, noise])
    points02, labels02 = gen_gauss([0.7, 0.8], 0, math.ceil(1/32*n_samples), [0.3*noise, 0.3*noise])
    points1, labels1 = gen_gauss([0.8, 0.7], 1, math.ceil(n_samples/2), [0.3*noise, 2*noise])
    points = points01 + points02 + points1
    labels = labels01 + labels02 + labels1

    return np.array(points, dtype=np.float32), labels


def spiral(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from two different noisy spirals.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Noise. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    n = math.ceil(n_samples/2)
    center = [0.5, 0.5]
    points0, labels0 = gen_spiral(0, 0, n, noise, center)
    points1, labels1 = gen_spiral(np.pi, 1, n, noise, center)
    points = points0 + points1
    labels = labels0 + labels1

    return np.array(points, dtype=np.float32), labels


def polynomial(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from two different noisy polynomials.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Noise. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    n = math.ceil(n_samples/2)
    points0, labels0 = gen_polynomial(0, n, noise)
    points1, labels1 = gen_polynomial(1, n, noise)
    points = points0 + points1
    labels = labels0 + labels1

    return np.array(points, dtype=np.float32), labels


def two_circle(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from two different noisy circles.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Noise. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    radius = 0.5
    center = [0.5, 0.5]
    n = math.ceil(n_samples/2)
    points0, labels0 = gen_circle(n, 0, 0, radius * 0.3, center)
    points1, labels1 = gen_circle(math.ceil(8/16*n_samples), 1, radius * 0.6, radius, center)
    points = points0 + points1
    labels = labels0 + labels1

    return np.array(points, dtype=np.float32), labels


def ellipse_circle(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from two different noisy circles.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Noise. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    radius = 0.5
    center = [0.5, 0.5]
    points0, labels0 = gen_gauss(center, 0, math.ceil(n_samples/2), [2.75*noise, 0.5 * noise])
    points1, labels1 = gen_circle(math.ceil(8/16 * n_samples), 1, radius * 0.7, radius, center)
    points = points0 + points1
    labels = labels0 + labels1

    return np.array(points, dtype=np.float32), labels


def three_yline(n_samples: int, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate a synthetic dataset consisting of points
    sampled from three different noisy lines along the y axis.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Noise. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    points01, labels01 = gen_yline(0, math.ceil(3/8*n_samples), 0.25, noise)
    points02, labels02 = gen_yline(0, math.ceil(1/8*n_samples), 0.75, noise)
    points1, labels1 = gen_yline(1, math.ceil(n_samples/2), 0.5, noise)
    points = points01 + points02 + points1
    labels = labels01 + labels02 + labels1

    return np.array(points, dtype=np.float32), labels


def circle_twogauss(n_samples: int, noise: float = 0.05):
    """Generate a synthetic dataset consisting of points
    sampled from one circle and two Gaussians inside the circle.

    Args:
        n_samples (int): Number of samples.
        noise (float, optional): Noise. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, List[int]]: Points, labels
    """
    radius = 0.5
    center = [0.5, 0.5]
    points01, labels01 = gen_gauss([0.35, 0.5], 0, math.ceil(n_samples/4), [noise/2, noise/2])
    points02, labels02 = gen_gauss([0.65, 0.5], 0, math.ceil(n_samples/4), [noise/2, noise/2])
    points1, labels1 = gen_circle(math.ceil(n_samples/2), 1, radius * 0.6, radius, center)
    points = points01 + points02 + points1
    labels = labels01 + labels02 + labels1

    return np.array(points, dtype=np.float32), labels


GENERATE_SYNTH = {
    'twogauss': two_gauss,
    'threegauss': three_gauss,
    'threegausso': three_gauss_o,
    'spiral': spiral,
    'polynomial': polynomial,
    'twocircle': two_circle,
    'threeyline': three_yline,
    'circletwogauss': circle_twogauss,
    'ellipsecircle': ellipse_circle
}


def generate_synth(dataset: str, n_samples: int = 100, noise: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Generate synthetic dataset.

    Args:
        name (str): Name of the synthetic dataset to generate.
        n_samples (int): Number of samples to generate.
        noise (float, optional): Standard deviation of noise to add.

    Returns:
        Tuple[np.ndarray, List[int]]: Data points and labels.
    """
    dataset = dataset.lower()
    if dataset in GENERATE_SYNTH:
        points, labels = GENERATE_SYNTH[dataset](n_samples, noise)
    else:
        raise ValueError(f'Error: unknown synthetic dataset {dataset}.')

    return points[:n_samples], labels[:n_samples]
