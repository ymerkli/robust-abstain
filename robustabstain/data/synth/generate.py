import numpy as np
from typing import Tuple, List, Union


def gen_gauss(
        mean: List[float], label: int, n_samples: int, noise: List[float]
    ) -> Tuple[List[List[float]], List[int]]:
    """Generate Gaussian distributed 2D points, according to distribution
    N((cx,cy), (noise, noise)).

    Args:
        mean (List[float]): Mean of Gaussian.
        label (int): Label of points.
        n_samples (int): Number of samples.
        noise (List[float]): Standard deviation of Gaussian.

    Returns:
        Tuple[List[List[float]], List[int]]: Points, labels
    """
    points, labels = [], []
    for _ in range(int(n_samples)):
        x = np.random.normal(mean[0], noise[0])
        y = np.random.normal(mean[1], noise[1])

        points.append([x,y])
        labels.append(label)

    return points, labels


def gen_spiral(
        deltaT: float, label: int, n_samples: int, noise: float, center: List[float]
    ) -> Tuple[List[List[float]], List[int]]:
    """Generate 2D points following a (noisy) spiral.

    Args:
        deltaT (float): theta offset.
        label (int): Label of points.
        n_samples (int): Number of samples.
        noise (float): Noise.
        center (List[float]): Center of the spirals.

    Returns:
        Tuple[List[List[float]], List[int]]: Points, labels
    """
    points, labels = [], []
    for i in range(n_samples):
        r = 0.5 * i / n_samples
        t = 1.5 * i / n_samples * 2 * np.pi + deltaT
        x = r * np.sin(t) + np.random.uniform(-0.2, 0.2) * noise + center[0]
        y = r * np.cos(t) + np.random.uniform(-0.2, 0.2) * noise + center[1]
        points.append([x,y])
        labels.append(label)

    return points, labels


def gen_polynomial(
        label: int, n_samples: int, noise: float
    ) -> Tuple[List[List[float]], List[int]]:
    """Generate 2D points following a (noisy) polynomial.

    Args:
        label (int): Label of points.
        n_samples (int): Number of samples.
        noise (float): Noise.

    Returns:
        Tuple[List[List[float]], List[int]]: Points, labels
    """
    points, labels = [], []
    def poly(x: float, version: int):
        if version == 0:
            y = 0.25 + 15 * x - 44 * x**2 - 73 * x**3 + 400 * x**4 - 474 * x**5 + 177 * x**6
        else:
            y = 1 + 13 * x - 44 * x**2 - 73 * x**3 + 400 * x**4 - 474 * x**5 + 177 * x**6
        return 0.5 * y

    for i in range(n_samples):
        x = i / n_samples
        y = poly(x, version=label) + np.random.uniform(-1, 1) * noise
        points.append([x,y])
        labels.append(label)

    return points, labels


def gen_circle(
        n_samples: int, label: int, radius_lb: float, radius_ub: float, center: List[float]
    ) -> Tuple[List[List[float]], List[int]]:
    """Generate 2D points following a (noisy) circle.

    Args:
        n_samples (int): Number of samples.
        label (int): Label of points.
        radius_lb (float): Lower bound of the radius.
        radius_ub (float): Upper bound of the radius.
        center (List[float]): Center of the circles..

    Returns:
        Tuple[List[List[float]], List[int]]: Points, labels
    """
    points, labels = [], []
    for i in range(n_samples):
        r = np.random.uniform(radius_lb, radius_ub)
        #r = np.random.normal((radius_ub-radius_lb)/2, 0.01)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.sin(theta) + center[0]
        y = r * np.cos(theta) + center[1]
        points.append([x,y])
        labels.append(label)

    return points, labels


def gen_ellipse(
        n_samples: int, label: int, rx: float, ry: float, center: List[float], noise: float
    ) -> Tuple[List[List[float]], List[int]]:
    """Generate 2D points following a (noisy) ellipse.

    Args:
        n_samples (int): Number of samples.
        label (int): Label of points.
        rx (float): Radius on x axis.
        ry (float): Radius on y ayis.
        center (List[float]): Center of the circles..
        noise (float): Noise.

    Returns:
        Tuple[List[List[float]], List[int]]: Points, labels
    """
    points, labels = [], []
    for i in range(n_samples):
        theta = np.random.uniform(0, 2*np.pi)
        x = rx * np.sin(theta) + center[0] + np.random.normal(0, noise)
        y = ry * np.cos(theta) + center[1] + np.random.normal(0, noise)
        points.append([x,y])
        labels.append(label)

    return points, labels


def gen_yline(
        label: int, n_samples: int, xpos: float, noise: float
    ) -> Tuple[List[List[float]], List[int]]:
    """Generate 2D points following a (noisy) line parallel to the y axis.

    Args:
        label (int): Label of points.
        n_samples (int): Number of samples.
        xpos (float): Line position on x axis.
        noise (float): Noise.

    Returns:
        Tuple[List[List[float]], List[int]]: Points, labels
    """
    points, labels = [], []
    for i in range(n_samples):
        x = np.random.normal(xpos, noise)
        y = np.random.uniform(0, 1)
        points.append([x,y])
        labels.append(label)

    return points, labels
