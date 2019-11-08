import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

p_0 = "C:\\Laboratory\\thesis\\Informs\\model\\pcd_0.ply"
p_1 = "C:\\Laboratory\\thesis\\Informs\\model\\pcd_1.ply"
p_2 = "C:\\Laboratory\\thesis\\Informs\\model\\pcd_2.ply"

pcd_0 = o3d.io.read_point_cloud(f"{p_0}")
pcd_1 = o3d.io.read_point_cloud(f"{p_1}")
pcd_2 = o3d.io.read_point_cloud(f"{p_2}")


pts_0 = np.asarray(pcd_0.points)
pts_1 = np.asarray(pcd_1.points)
pts_2 = np.asarray(pcd_2.points)

fig = plt.figure(figsize=())
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(pts_0[:,0], pts_0[:,1], pts_0[:,2], s=0.2, c='r')
ax.scatter(pts_1[:,0], pts_1[:,1], pts_1[:,2], s=0.2, c='g')
ax.scatter(pts_2[:,0], pts_2[:,1], pts_2[:,2], s=0.2, c='b')
plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

X = np.concatenate([pts_0[:,0], pts_1[:,0], pts_2[:,0]])
Y = np.concatenate([pts_0[:,1], pts_1[:,1], pts_2[:,1]])
Z = np.concatenate([pts_0[:,2], pts_1[:,2], pts_2[:,2]])
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.3

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# plt.show()
plt.savefig('trees.png')
# o3d.visualization.draw_geometries([pcd])