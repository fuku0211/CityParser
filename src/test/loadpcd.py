import argparse
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()


pcd = o3d.io.read_point_cloud(f"./{args.file}.pcd")
o3d.visualization.draw_geometries([pcd])