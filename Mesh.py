import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

import open3d

class Mesh:

    def __init__(self, file_path):
        """
        Initializes mesh by reading it in as point cloud and then downsamples
        and normalizes it

        Args:
            file_path: path to STL file of skull

        Returns:
            None
        """
        self.voxel_size = 0.15

        # read point cloud from STL file and downsample
        triangle_mesh = open3d.io.read_triangle_mesh(file_path)
        pcd = open3d.geometry.PointCloud()
        pcd.points = triangle_mesh.vertices
        pcd.colors = triangle_mesh.vertex_colors
        pcd.normals = triangle_mesh.vertex_normals
        pcd = Mesh.normalize(pcd)
        self.skull = pcd.voxel_down_sample(self.voxel_size)
        self.skull = Mesh.normalize(self.skull)
        self.skull.estimate_normals(search_param=
            open3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2,
            max_nn=30))

        self.vis = open3d.visualization.VisualizerWithEditing()

    def align_with_z_axis(self):
        """
        Aligns skull model to +z axis to be right-side up

        Args:
            None

        Returns:
            None
        """
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
            open3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2,
            max_nn=30))

    def crop_geometry(self):
        """
        User determines geometry to crop from original point cloud. Those points
        are subtracted from original point cloud to create defect. The cropped
        point cloud is transformed to different poses and projected onto
        original point cloud to generate more skull defect samples. Label
        matrices are also created, indicating which points are associated to
        defect contour.

        Args:
            pcd: point cloud

        Returns:
            skull defect point cloud samples
            label matrix for each skull defect point cloud sample
        """
        pass

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
        # - for each point in rotated defect pcd, find closest point on total
        #   point cloud
        # - subtract closest points out

        # How label points as defect after subtraction
        # - extract boundary of rotated defect point cloud
        # - for each point on boundary of rotated defect point cloud, find
        #   closest point on total point cloud
        """

    @staticmethod
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

    def pick_top_point(self):
        """
        Lets user select top point of skull such that said point will be lie
        along +z axis

        Args:
            None

        Returns:
            None
        """
        # let user pick point on top of skull
        print("Shift+click to select point approximately at top of skull. Exit window when done.\n")
        self.vis.create_window()
        self.vis.add_geometry(self.skull)
        self.vis.run()
        self.vis.destroy_window()

    def remove_inner_layer(self):
        """
        Remove inner layer of skull model to only obtain outer surface

        Args:
            None

        Returns:
            None
        """
        diameter = np.linalg.norm(np.asarray(self.skull.get_max_bound()) -
            np.asarray(self.skull.get_min_bound()))
        camera = [ [0, -diameter, 0],
                   [0, diameter, 0],
                   [-diameter, 0, 0],
                   [diameter, 0, 0],
                   [0, 0, diameter] ]
        radius = diameter * 100

        pts = []
        surface = open3d.geometry.PointCloud()
        for i in range(len(camera)):
            pcd, _ = self.skull.hidden_point_removal(camera[i], radius)
            pts.extend(pcd.vertices)
        surface.points = open3d.utility.Vector3dVector(np.asarray(pts))
        trimesh = open3d.geometry.TriangleMesh()
        trimesh.vertices = surface.points
        trimesh = trimesh.remove_duplicated_vertices()
        surface.points = trimesh.vertices
        open3d.visualization.draw_geometries([surface])

    def visualize_mesh_with_matplotlib(self):
        """
        Visualize skull model using matplotlib instead of open3d

        Args:
            None

        Returns:
            None
        """
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


m = Mesh("./STL/CRANIAL HEADS_Head_1_001.stl")
m.remove_inner_layer()
