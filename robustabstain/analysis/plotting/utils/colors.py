from typing import List


def convert_rgb(color: List[int]) -> str:
    """Convert RGB color array to matplotlib rgb string.

    Args:
        color (List[int]): RGB color array.

    Returns:
        str: RGB color string usable in matplotlib.
    """
    assert len(color) == 3, 'Error: RGB color must consist of 3 integers.'
    float_color = [val/255 for val in color]
    return f"rgb({float_color[0]}, {float_color[1]}, {float_color[2]})"


def normalize_color(color: List[int]) -> List[float]:
    """Convert RGB color to normalized RGB color.

    Args:
        color (List[int]): RGB color array.

    Returns:
        List[float]: Normalized RGB color. 
    """
    assert len(color) == 3, 'Error: RGB color must consist of 3 integers.'
    return [val/255 for val in color] 


def convert_color_pairs(color_pairs: List[List[int]]) -> List[List[float]]:
    """Convert RGB colors in color pairs to rgb string colors.

    Args:
        color_pairs (List[List[int]]): Color pairs.

    Returns:
        List[List[str]]: Converted colorpairs for matplotlib.
    """
    for i in range(len(color_pairs)):
        color_0 = tuple(normalize_color(color_pairs[i][0]))
        color_1 = tuple(normalize_color(color_pairs[i][1]))
        color_pairs[i] = [color_0, color_1]
    return color_pairs


# pairs of named RGB colors for matplotlib that have a similar color and shade
COLOR_PAIRS_1 = [
    [[77, 38, 0], [128, 64, 0]], # brown
    [[255, 128, 0], [255, 179, 102]], # orange
    [[179, 0, 0], [255, 153, 153]], # red
    [[255, 77, 210], [255, 179, 236]], # pink
    [[102, 0, 102], [204, 0, 204]], # purple
    [[0, 0, 0], [77, 77, 77]] # black/grey 
]
COLOR_PAIRS_2 = [
    [[0, 102, 0], [0, 204, 0]], # darkgreen
    [[0, 102, 153], [0, 170, 255]], # turquoise
    [[0, 0, 153], [26, 26, 255]] # darkblue
]

COLOR_PAIRS_1 = convert_color_pairs(COLOR_PAIRS_1)
COLOR_PAIRS_2 = convert_color_pairs(COLOR_PAIRS_2)


LF_COLORS = {
    'blue': normalize_color([49, 142, 222]),
    'orange': normalize_color([255, 158, 25])
}

LF_COLORS_LIGHT = {
    'blue': normalize_color([145, 200, 250]),
    'orange': normalize_color([255, 205, 135])
}


# robustness / accuraccy colors
ROBACC_COLORS = {
    'ra': normalize_color([0, 90, 181]),
    'nra': normalize_color([153, 204, 255]),
    'ria': normalize_color([179, 41, 25]),
    'nria': normalize_color([242, 173, 166])
}