import colorsys
import math
import random
import numpy as np
from geometry.capture import WIDTH, HEIGHT
import cupy as cp
import numba


def calc_gap_between_yaxis_and_vector(x, y):
    t = math.atan2(y, x)
    s = None
    if y >= 0:
        # 第一象限
        if x >= 0:
            s = math.pi / 2 - t
        # 第二象限
        else:
            s = -t + math.pi / 2
    else:
        # 第三象限
        if x < 0:
            s = -3 / 2 * math.pi - t
        # 第四象限
        else:
            s = -3 / 2 * math.pi - t

    return s


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def array_to_3dim(array):
    """hdf5に保存した1次元の画像配列を3次元か2次元配列に復元する

    Parameters
    ----------
    array : numpy array
        hdf5から読み込んだ配列

    Returns
    -------
    ndarray
        shape = (横解像度、縦解像度、n)の配列
        n はrgbのときは3

    """

    if array.size == WIDTH * HEIGHT * 3:
        result = np.asarray(array).reshape([HEIGHT, WIDTH, 3])
    else:
        result = np.asarray(array).reshape([HEIGHT, WIDTH])
    return result
