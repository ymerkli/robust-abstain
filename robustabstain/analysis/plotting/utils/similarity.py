import pandas as pd
import numpy as np
from typing import List


def smc(a: np.ndarray, b: np.ndarray) -> float:
    """Simple matching coefficient (considers both presence and absence (i.e. both (0,0) and (1,1)) as similarity)
    """
    assert len(a) == len(b)
    n = len(a)
    return (a == b).sum()/n


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity coefficient (only considers presence (i.e. (1,1)) as similarity)
    """
    assert len(a) == len(b)
    n = (a == 1).sum() + (b == 1).sum() - ((a == 1) & (b == 1)).sum()
    js = ((a == 1) & (b == 1)).sum()/n if n != 0 else 0
    return js


def jaccard_similarity0(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity coefficient (only considers absence (i.e. (0,0)) as similarity)
    """
    assert len(a) == len(b)
    n = (a == 0).sum() + (b == 0).sum() - ((a == 0) & (b == 0)).sum()
    js = ((a == 0) & (b == 0)).sum()/n if n != 0 else 0
    return js


def df_jaccard_similarity(df: pd.DataFrame, cols: List[str] = None) -> float:
    """Jaccard similarity coefficient over all given columns of a given DataFrame
    """
    if cols is None:
        cols = df.columns
    # capture all rows where at least one column from cols is equal to 1
    index_or = [False] * len(df)
    # capture all rows where all columns from cols are equal to 1
    index_and = [True] * len(df)
    for col_name in cols:
        index_or |= df[col_name] == 1
        index_and &= df[col_name] == 1
    js = len(df[index_and]) / len(df[index_or]) if len(df[index_or]) != 0 else 0
    return js


def hamann_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hamman similarity coefficient
    """
    assert len(a) == len(b)
    n_attr = len(a)
    return ((a == b).sum() - (a != b).sum())/n_attr
