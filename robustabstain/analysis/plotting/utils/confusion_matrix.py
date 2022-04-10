import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List


def plot_confusion_matrix(
        cm: np.ndarray, target_names: List[str], title='Confusion matrix',
        cmap: matplotlib.colors.Colormap = None, normalize: bool = True,
        savepath: str = None
    ) -> None:
    """Plot a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix procuced by sklearn.metrics.confusion_matrix
        target_names (List[str]): Label names.
        title (str, optional): Plot title. Defaults to 'Confusion matrix'.
        cmap (matplotlib.colors.Colormap, optional): Colormap. Defaults to None.
        normalize (bool, optional): Normalize counts. Defaults to True.
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('YlOrBr')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    if savepath is not None:
        plt.savefig(savepath+'.png', dpi=300,  bbox_inches="tight")
        plt.savefig(savepath+'.pdf', bbox_inches="tight")