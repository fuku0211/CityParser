import colorsys
import math
import random
import numpy as np
from geometry.capture import WIDTH, HEIGHT
import cupy as cp
import numba
import subprocess
import json

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]


def calc_angle_between_axis(vector, axis="x"):
    if axis == "x":
        axis = np.array([0, 1])
        angle = np.arccos(
            np.dot(axis, vector) / (np.linalg.norm(axis) * np.linalg.norm(vector))
        )
        if vector[0] < 0:
            angle = 2 * math.pi - angle

        return angle


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
        result = np.asanyarray(array).reshape([HEIGHT, WIDTH, 3])
    else:
        result = np.asanyarray(array).reshape([HEIGHT, WIDTH])
    return result
