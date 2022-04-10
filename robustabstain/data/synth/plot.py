import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable


def plot_synth(points: np.ndarray, labels: List[int], savepath: str = None):
    color = {0: 'blue', 1: 'orange', 2: 'green'}
    fig, ax = plt.subplots()
    for point, label in zip(points, labels):
        ax.scatter(point[0], point[1], c=color[label])

    if savepath:
        savepath = Path(savepath)
        if not os.path.isdir(savepath.parent.absolute()):
            os.makedirs(savepath.parent.absolute())

        fig.tight_layout()
        fig.savefig(str(savepath)+'.png', dpi=300)
        fig.savefig(str(savepath)+'.pdf')

    plt.show()