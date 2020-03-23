import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

import open3d

class Mesh:

    def __init__(self, file_path):
        # read point cloud from STL file and downsample
        triangle_mesh = open3d.io.read_triangle_mesh(file_path)
        pcd = open3d.geometry.PointCloud()
        pcd.points = triangle_mesh.vertices
        pcd.colors = triangle_mesh.vertex_colors
        pcd.normals = triangle_mesh.vertex_normals
        pcd = normalize(pcd)
        self.skull = pcd.voxel_down_sample(voxel_size=0.15)
        self.skull = normalize(self.skull)

        self.vis = open3d.visualization.VisualizerWithEditing()

    def align_with_z_axis(self):
        PickTopPoint()

        # normalize vector associated with point on top of skull
        top_idx = self.vis.get_picked_points()[-1]
        top = np.asarray(self.skull.points[top_idx])
        top = top / np.linalg.norm(top)

        # calculate rotation to orient vector to top of skull to +z axis
        z = np.asarray([0, 0, 1])
        theta = np.arccos(np.dot(top, z))
        rvec = np.cross(top, z)
        rvec = rvec / np.linalg.norm(rvec)
        rot = R.from_rotvec(theta * rvec)
        rot_mat = rot.as_matrix()

        # rotate
        downpcd_pts = np.asarray(self.skull.points).T 
        rot_pts = np.matmul(rot_mat, downpcd_pts)

        # assign to new point cloud
        self.skull.points = open3d.utility.Vector3dVector(rot_pts.T)
        self.skull.colors = downpcd.colors
        self.skull.estimate_normals(search_param=
            open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # open3d.visualization.draw_geometries([self.skull])

    def crop_geometry(self):
        """
        User determines geometry to crop from original point cloud. Those points are
        subtracted from original point cloud to create defect. The cropped point
        cloud is transformed to different poses and projected onto original point
        cloud to generate more skull defect samples. Label matrices are also created,
        indicating which points are associated to defect contour.

        Args:
            pcd: point cloud

        Returns:
            skull defect point cloud samples
            label matrix for each skull defect point cloud sample
        """
        print("Demo for manual geometry cropping")
        print(
            "1) Press 'Y' twice to align geometry with negative direction of y-axis"
        )
        print("2) Press 'K' to lock screen and to switch to selection mode")
        print("3) Drag for rectangle selection,")
        print("   or use ctrl + left click for polygon selection")
        print("4) Press 'C' to get a selected geometry and to save it")
        print("5) Press 'F' to switch to freeview mode")
        open3d.visualization.draw_geometries_with_editing([self.skull])

        # save cropped point cloud
        # load cropped point cloud
        # filter out bottom half of points b/c defect is only on top of skull

        # How rotate defect to other poses and extract those points?
        # - for each point in rotated defect pcd, find closest point on total point
        #   cloud
        # - subtract closest points out

        # How label points as defect after subtraction
        # - extract boundary of rotated defect point cloud
        # - for each point on boundary of rotated defect point cloud, find closest
        #   point on total point cloud

    def pick_top_point(self):
        # let user pick point on top of skull
        print("Shift+click to select point approximately at top of skull. Exit window when done.\n")
        self.vis.create_window()
        self.vis.add_geometry(self.skull)
        self.vis.run()
        self.vis.destroy_window()

    def remove_inner_layer(self):
        pass

    def visualize_mesh_with_matplotlib(self):
        # visualize rotated point cloud with matplotlib
        pts = np.asarray(self.skull.points)
        # print(len(pts))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]
        ax.scatter(x[::3], y[::3], z[::3])
        ax.set_xlabel('Normalized x')
        ax.set_ylabel('Normalized y')
        ax.set_zlabel('Normalized z')
        plt.show()
