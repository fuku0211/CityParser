import colorsys
import math
import random
import subprocess
from pprint import pprint

import numpy as np
from scipy.spatial import voronoi_plot_2d

from geometry.capture import HEIGHT, WIDTH
from utils.color_output import output_with_color
from inputimeout import inputimeout, TimeoutOccurred

DEFAULT_ATTRIBUTES = (
    "index",
    "uuid",
    "name",
    "timestamp",
    "memory.total",
    "memory.free",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
)


def get_gpu_info(nvidia_smi_path="nvidia-smi", keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = "" if not no_units else ",nounits"
    cmd = "%s --query-gpu=%s --format=csv,noheader%s" % (
        nvidia_smi_path,
        ",".join(keys),
        nu_opt,
    )
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]

    return [{k: v for k, v in zip(keys, line.split(", "))} for line in lines]


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
    """hdf5に保存した1次元の画像配列を3次元か2次元に復元する

    Args:
        list(int): hdf5上のflatten化された画像配列

    Returns:
        np.ndarray: (横解像度、縦解像度、n)の配列
    """
    if array.size == WIDTH * HEIGHT * 3:
        result = np.asanyarray(array).reshape([HEIGHT, WIDTH, 3])
    else:
        result = np.asanyarray(array).reshape([HEIGHT, WIDTH])
    return result


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if set(v) == set(val)]
    return keys


def plot_voronoi_with_label(vor):
    fig = voronoi_plot_2d(vor)
    for i in range(vor.points.shape[0]):
        fig.axes[0].text(vor.points[i, 0], vor.points[i, 1], f"{i}")
    for i in range(vor.vertices.shape[0]):
        fig.axes[0].text(vor.vertices[i, 0], vor.vertices[i, 1], f"{i}")
    fig.axes[0].set_aspect("equal")
    fig.show()


def select_cluster_setting(site_path, instead):
    """点群の設定を選択してファイル名を返却する

    Parameters
    ----------
    site_path : Path
        敷地のptsフォルダ

    Returns
    -------
    file_name : str
        segmentation
    """
    output_with_color("select setting", "g")
    # 点群設定の一覧を出力する
    file_names = [i.name for i in site_path.glob("clustering_*.pts")]
    tmp = list(zip(range(len(file_names)), file_names))
    pprint(dict(tmp))

    # 入力した数値を元に設定を選択する
    try:
        file_name = file_names[int(inputimeout(prompt="select setting by index \n>", timeout=3))]
    except TimeoutOccurred:
        print("timeout")
        file_name = file_names[instead]
    print(f"selected : {file_name}")
    return file_name
