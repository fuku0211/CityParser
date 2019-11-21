import colorsys
import math
import random
import subprocess
from operator import itemgetter

import numpy as np
import pyproj
from geometry.capture import HEIGHT, WIDTH

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


# TRANSFORMER = pyproj.Transformer.from_crs("EPSG:6668", "EPSG:30169")
TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:30169")


def parse_gps_data(gpsdata):
    """gpsのデータを変換して取り出す

    Args:
        gpsdata (list[str]): dgpro-1rwで取得したgpsデータ

    Returns:
        tuple: (x座標, y座標, 方向, 高度)

    Note:
        座標は平面直角座標の投影された値
        方向は北を0として右回りで計測した値
    """
    # TODO: 標高=楕円体高であってるかわからない
    lat, lon, dire, ht = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
    # 欠損値に対する処理
    if lat < 0 or lon < 0:
        x, y = None, None
    # dddmm.mmmm表記になっているのを(度数+分数/60)でddd.dddd表記にする
    # http://lifelog.main.jp/wordpress/?p=146
    else:
        dd_lat, mm_lat = divmod(lat / 100, 1)
        dd_lon, mm_lon = divmod(lon / 100, 1)
        lat = dd_lat + mm_lat * 100 / 60
        lon = dd_lon + mm_lon * 100 / 60
        y, x = TRANSFORMER.transform(lat, lon)
    return (x, y, dire, ht)
