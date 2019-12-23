import colorsys
import math
import random
import subprocess
from operator import itemgetter
from pprint import pprint

import numpy as np
import pyproj
from scipy.spatial import voronoi_plot_2d

from geometry.capture import HEIGHT, WIDTH
from utils.color_output import output_with_color

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


TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:30169")
# TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6677")


def parse_x_y_from_gps(gpsdata):
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


def parse_lat_lon_from_gps(gpsdata):
    """gpsのデータを変換して緯度と経度を取り出す

    Args:
        gpsdata (list[str]): dgpro-1rwで取得したgpsデータ

    Returns:
        緯度、軽度

    Note:
        座標は平面直角座標の投影された値
        方向は北を0として右回りで計測した値
    """
    # TODO: 標高=楕円体高であってるかわからない
    lat, lon, dire, ht = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
    # 欠損値に対する処理
    if lat < 0 or lon < 0:
        lat, lon = None, None
    # dddmm.mmmm表記になっているのを(度数+分数/60)でddd.dddd表記にする
    # http://lifelog.main.jp/wordpress/?p=146
    else:
        dd_lat, mm_lat = divmod(lat / 100, 1)
        dd_lon, mm_lon = divmod(lon / 100, 1)
        lat = dd_lat + mm_lat * 100 / 60
        lon = dd_lon + mm_lon * 100 / 60
    return lat, lon


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


def select_pcd_setting(site_path):
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
    file_name = file_names[int(input("select setting by index \n>"))]
    print(f"selected : {file_name}")

    return file_name
