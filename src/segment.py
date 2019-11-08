import argparse
from pathlib import Path
from time import sleep

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# コマンドライン用
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["segment", "play"], help="execution mode")
parser.add_argument("-s", "--record_start")
parser.add_argument("-e", "--record_end")
parser.add_argument("-v", "--view")
args = parser.parse_args()

# DIR_RECORD = '/content/drive/My Drive/Colab Notebooks/distro_analyzer/out/record'
#
DIR_RECORD = "C:/Laboratory/model/distro_analyzer/out/record"


def __create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

    Returns:
    A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """

    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = __create_cityscapes_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


if args.mode == "segment":
    from geometry.model import create_model, deeplabv3

    if args.record_start is None or args.record_end is None:
        raise NoRangeError("specify start and end date")

    # コマンドラインで指定した範囲の録画のパスを取り出す
    records = list(Path(DIR_RECORD).iterdir())
    dates = sorted([i.stem for i in records])
    records = records[
        dates.index(f"{args.record_start}") : dates.index(f"{args.record_end}") + 1
    ]

    model = create_model()
    for record in records:
        print("record : " + record.stem)
        path_color = Path(record) / Path(record.stem + "_color.npy")
        npy = np.load(path_color)
        frames, height, width, _ = npy.shape

        # セグメンテーション結果を格納する配列
        seg = np.empty((frames, height, width), dtype="uint8")
        for i in tqdm(range(frames)):
            # フレームを読み込んでセグメンテーションできるようにリサイズする
            img = Image.fromarray(npy[i, :, :, :])
            img = img.resize((2048, 1024))
            # セグメンテーションして元のサイズに戻す
            result = deeplabv3(img, model).astype("uint8")
            result = result == 8
            result = result * 255
            result = np.dstack([result, result, result]).astype("uint8")
            result = Image.fromarray(result).resize((width, height))
            result = np.asarray(result)
            tag = result[:, :, 0] + result[:, :, 1] + result[:, :, 2]
            tree = tag != 0
            seg[i, :, :] = tree
        path_seg = Path(record) / Path(record.stem + "_segment.npy")
        np.save(path_seg, seg)

if args.mode == "play":

    if args.view is None:
        raise NoViewFile("segment view file")

    path_seg = Path(DIR_RECORD) / Path(args.view) / Path(f"{args.view}_segment.npy")
    path_org = Path(DIR_RECORD) / Path(args.view) / Path(f"{args.view}_color.npy")
    seg = np.load(path_seg)
    org = np.load(path_org)

    for i in range(seg.shape[0]):
        # レンダリング
        seg_frame = seg[i, :, :]
        # boolをintにして*8することで植物の部分が8になるようにする
        seg_frame = seg_frame * 8
        seg_frame = label_to_color_image(seg_frame)

        seg_frame = Image.fromarray(seg_frame)
        org_frame = Image.fromarray(org[i,:,:])
        mix_frame = Image.blend(org_frame, seg_frame, 0.7)
        mix_frame = cv2.cvtColor(np.asarray(mix_frame), cv2.COLOR_BGR2RGB)
        cv2.imshow("RealSense", mix_frame)
        key = cv2.waitKey(1) & 0xFF

        sleep(0.1)

        # qキーを押したら終了
        if key == ord("q"):
            cv2.destroyAllWindows()
            cv2.imwrite('frame.jpg', mix_frame)
            exit()
