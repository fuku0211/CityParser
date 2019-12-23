import h5py
from pathlib import Path
from contextlib import ExitStack
import datetime
from tqdm import tqdm

site_name = "nezu_2"
file_dir = Path("data", "hdf5", site_name)
color_path = file_dir / Path("color.hdf5")
depth_path = file_dir / Path("depth.hdf5")
gps_path = file_dir / Path("gps.hdf5")
seg_path = file_dir / Path("seg.hdf5")

base_date = "20191211_125844"
add_date = "20191215_150159"
add_pt = 85
t_start = datetime.datetime.now()

def _create_zfill_time(time_now):
    """ソートしやすいようゼロ埋めした日時を返却する

    Parameters
    ----------
    time_now : datetime.datetime
        現在日時

    Returns
    -------
    string
        yymmdd_hhmmss
    """
    y = time_now.year
    mo = str(time_now.month).zfill(2)
    d = str(time_now.day).zfill(2)
    h = str(time_now.hour).zfill(2)
    mi = str(time_now.minute).zfill(2)
    se = str(time_now.second).zfill(2)
    return f"{y}{mo}{d}_{h}{mi}{se}"

with ExitStack() as stack:
    fc = stack.enter_context(h5py.File(color_path, "a"))
    fd = stack.enter_context(h5py.File(depth_path, "a"))
    fg = stack.enter_context(h5py.File(gps_path, "a"))
    fs = stack.enter_context(h5py.File(seg_path, "a"))
    srcs = [fc, fd, fg, fs]

    for src in srcs:
        new_group = src.create_group(_create_zfill_time(t_start))
        base_group = src[base_date]
        add_group = src[add_date]
        for i in tqdm(base_group.keys()):
            new_group.create_dataset(i, data=base_group[i], compression="gzip")
        max_idx = max(map(int, list(base_group.keys())))
        for i, key in enumerate(tqdm(add_group.keys())):
            if int(i) >= add_pt:
                new_group.create_dataset(str(max_idx + i + 1), data=add_group[key], compression="gzip")




