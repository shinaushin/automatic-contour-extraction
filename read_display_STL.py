import numpy as np
import os

import open3d

# visualize every STL file's downsampled point cloud
for file in os.listdir("./STL"):
    if file.endswith(".stl"):
        skull_file = os.path.join("./STL", file)

        skull = open3d.io.read_triangle_mesh(skull_file)

        pcd = open3d.geometry.PointCloud()
        pcd.points = skull.vertices
        pcd.colors = skull.vertex_colors
        pcd.normals = skull.vertex_normals
        center = pcd.get_center()
        pts = np.asarray(pcd.points)
        stddev = [np.std(pts[:,0]), np.std(pts[:,1]), np.std(pts[:,2])]
        pts = (pts - center) / stddev
        pcd.points = open3d.utility.Vector3dVector(pts)

        downpcd = pcd.voxel_down_sample(voxel_size=0.15)
        downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
        print(np.asarray(downpcd.points).shape)
        print("---")

        open3d.visualization.draw_geometries([downpcd])
