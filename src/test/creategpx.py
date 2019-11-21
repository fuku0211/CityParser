import gpxpy
import gpxpy.gpx
import h5py
from operator import itemgetter

gpx = gpxpy.gpx.GPX()

# Create first track in our GPX:
gpx_track = gpxpy.gpx.GPXTrack(name="test")
gpx.tracks.append(gpx_track)

# Create first segment in our GPX track:
gpx_segment_a = gpxpy.gpx.GPXTrackSegment()
gpx_segment_b = gpxpy.gpx.GPXTrackSegment()
gpx_track.segments.append(gpx_segment_a)
gpx_track.segments.append(gpx_segment_b)

px_file_w = open("sample_timefix.gpx", "w")


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
        lat, lon = None, None
    # dddmm.mmmm表記になっているのを(度数+分数/60)でddd.dddd表記にする
    # http://lifelog.main.jp/wordpress/?p=146
    else:
        dd_lat, mm_lat = divmod(lat / 100, 1)
        dd_lon, mm_lon = divmod(lon / 100, 1)
        lat = dd_lat + mm_lat * 100 / 60
        lon = dd_lon + mm_lon * 100 / 60
    return (lat, lon)


# Create points:
with h5py.File(
    "C:\\Laboratory\\model\\distro_analyzer\\data\\hdf5\\test\\gps.hdf5", "r"
) as f:
    group_a = f["20191112_142615_a"]
    group_b = f["20191112_142615_b"]

    for i in range(len(group_a.keys())):
        lat, lon = parse_gps_data(group_a[str(i)])
        gpx_segment_a.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))
    for i in range(len(group_b.keys())):
        lat, lon = parse_gps_data(group_b[str(i)])
        gpx_segment_b.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

# You can add routes and waypoints, too...
px_file_w.write(gpx.to_xml(version="1.1"))
px_file_w.close()
