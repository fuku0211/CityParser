import argparse
import open3d as o3d

# parser = argparse.ArgumentParser()
# parser.add_argument("file")
# args = parser.parse_args()


pcd = o3d.io.read_point_cloud("src/test/binary/cloud_bin_0.pcd")
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=[0, 0, 0]
)
# pcd.paint_uniform_color([1, 0.706, 0])
pcd1 = pcd.rotate([0,0,0], center=False)
o3d.visualization.draw_geometries([pcd1, mesh_frame])