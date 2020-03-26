import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

import open3d

def normalize(pcd):
    """
    Normalize point cloud, mean of 0, stddev of 1
    
    Args:
        pcd: Open3D point cloud object

    Returns:
        normalized pointcloud
    """
    center = pcd.get_center()
    pts = np.asarray(pcd.points)
    stddev = [np.std(pts[:,0]), np.std(pts[:,1]), np.std(pts[:,2])]
    pts = (pts - center) / stddev
    pcd.points = open3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

for f in os.listdir("./STL"):
    if f.endswith(".stl"):
        skull_file = os.path.join("./STL", f)
        print(skull_file)
        # read point cloud from STL file and downsample
        skull = open3d.io.read_triangle_mesh(skull_file)
        pcd = open3d.geometry.PointCloud()
        pcd.points = skull.vertices
        pcd.colors = skull.vertex_colors
        pcd.normals = skull.vertex_normals
        pcd = normalize(pcd)
        downpcd = pcd.voxel_down_sample(voxel_size=0.15)
        downpcd = normalize(downpcd)

        # cannot use VisualizerWithEditing, close window, and then display next STL file
        # have no idea why
        """
        # let user pick point on top of skull
        print("Shift+click to select point approximately at top of skull. Exit window when done.\n")
        vis = open3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(downpcd)
        vis.run()
        vis.destroy_window()
        
        # normalize vector associated with point on top of skull
        top_idx = vis.get_picked_points()[-1]
        top = np.asarray(downpcd.points[top_idx])
        top = top / np.linalg.norm(top)

        # calculate rotation to orient vector to top of skull to +z axis
        z = np.asarray([0, 0, 1])
        theta = np.arccos(np.dot(top, z))
        rvec = np.cross(top, z)
        rvec = rvec / np.linalg.norm(rvec)
        rot = R.from_rotvec(theta * rvec)
        rot_mat = rot.as_matrix()

        # rotate
        downpcd_pts = np.asarray(downpcd.points).T 
        rot_pts = np.matmul(rot_mat, downpcd_pts)

        # assign to new point cloud
        rot_pcd = open3d.geometry.PointCloud()
        rot_pcd.points = open3d.utility.Vector3dVector(rot_pts.T)
        rot_pcd.colors = downpcd.colors
        rot_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        trimesh = open3d.geometry.TriangleMesh()
        trimesh.vertices = rot_pcd.points
        trimesh = trimesh.compute_vertex_normals()
        trimesh = trimesh.compute_triangle_normals()
        new_file = os.path.join("./Rotated_STL", f[:-4])
        # open3d.io.write_triangle_mesh("hi.ply", trimesh)
        """

# # open3d.visualization.draw_geometries([rot_pcd])

# # visualize rotated point cloud with matplotlib
# pts = np.asarray(rot_pcd.points)
# print(len(pts))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = pts[:,0]
# y = pts[:,1]
# z = pts[:,2]
# ax.scatter(x[::3], y[::3], z[::3])
# ax.set_xlabel('Normalized x')
# ax.set_ylabel('Normalized y')
# ax.set_zlabel('Normalized z')
# plt.show()
