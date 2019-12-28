import pyproj
from operator import itemgetter
import math
from pathlib import Path
import urllib.request

TRANSFORMER_TO_XY = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:30169")
# TRANSFORMER_TO_XY = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6677")
TRANSFORMER_TO_LAT_LON = pyproj.Transformer.from_crs("EPSG:30169", "EPSG:4326")


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
        ht = get_elevation_from_tile(lat, lon)
        y, x = TRANSFORMER_TO_XY.transform(lat, lon)
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
    lat, lon, _, _ = map(float, itemgetter(3, 5, 8, 33)(gpsdata))
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


def convert_xy_to_latlon(x, y):
    lat, lon = TRANSFORMER_TO_LAT_LON.transform(y, x)
    return lat, lon


def get_elevation_from_tile(lat, lon, zoom=15, dem_src="dem5a", data_round=1):
    """緯度経度から標高を返却する

    Parameters
    ----------
    lat : float
        緯度
    lon : float
        経度
    zoom : int, optional
        タイルのズーム, by default 15
    dem_src : str, optional
        計測方法, by default "dem5a"
    data_round : int, optional
        data_round, by default 1
    save_dir : Path, optional
        タイルのtxtファイルを保存するディレクトリ, by default None

    Returns
    -------
    elevation : float
        標高

    Notes
    -------
    国土地理院の標高タイルを利用
    """
    def _get_world_coords(lat, lon):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        r = 128 / math.pi
        coord_x = r * (lon_rad + math.pi)
        coord_y = (
            -1 * r / 2 * math.log((1 + math.sin(lat_rad)) / (1 - math.sin(lat_rad)))
            + 128
        )
        return coord_x, coord_y

    def _get_elevation(coord_x, coord_y, zoom, dem_src, data_round, save_path=None):
        pixel_x = coord_x * math.pow(2, zoom)
        tile_x = math.floor(pixel_x / 256)
        pixel_y = coord_y * math.pow(2, zoom)
        tile_y = math.floor(pixel_y / 256)

        pixel_x_int = math.floor(pixel_x)
        px = pixel_x_int % 256
        pixel_y_int = math.floor(pixel_y)
        py = pixel_y_int % 256


        # 標高タイルの読み込み
        file_name = f"{dem_src}_{zoom}_{tile_x}_{tile_y}.txt"
        save_dir = Path("data", "tile")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, file_name)
        try:
            with open(save_path) as f:
                text_raw = f.readlines()
                return float(text_raw[py].split(",")[px])

        except FileNotFoundError:
            # 標高タイルをダウンロード
            try:
                base_url = "http://cyberjapandata.gsi.go.jp/xyz/"
                file_url = file_name.replace("_", "/")
                urllib.request.urlretrieve(base_url + file_url, str(save_path))
                with open(save_path) as f:
                    text_raw = f.readlines()
                    return float(text_raw[py].split(",")[px])

            except urllib.error.HTTPError:
                print("httperror : {file}")
                exit()

    coord_x, coord_y = _get_world_coords(lat, lon)
    return _get_elevation(coord_x, coord_y, zoom, dem_src, data_round)
