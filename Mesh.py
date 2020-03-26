import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
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

    def extract_labeled_data(self):
        """
        Creates approximate defect, labels defect contour points as separate
        class from rest of skull points. Generates 15x15x15 matrix per defect as
        training data input for DL model

        Args:
            pcd: point cloud

        Returns:
            skull defect point cloud samples
            label matrix for each skull defect point cloud sample
        """
        diameter = np.linalg.norm(np.asarray(self.skull.get_max_bound()) -
            np.asarray(self.skull.get_min_bound()))
        
        camera = [diameter/3.5, diameter/3.5, diameter]
        radius = diameter/2.15
        camera_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
            origin=camera)

        defect = open3d.geometry.PointCloud()
        defect_pcd, _ = self.skull.hidden_point_removal(camera, radius)
        defect_vertices = [list(pt) for pt in defect_pcd.vertices if pt[2] > 0.20]
        defect_indices = [i for i in range(len(defect_pcd.vertices)) if defect_pcd.vertices[i][2] > 0.20]
        defect.points = open3d.utility.Vector3dVector(np.asarray(defect_vertices))

        # project points with +z value onto xy plane 
        defect_on_xy = np.asarray(defect_vertices)[:,0:2]

        # find points that form convex hull of defect using scipy
        hull = ConvexHull(defect_on_xy)

        # Smooth out convex hull (TODO)
        # xhull = defect_vertices[hull.vertices,0]
        # yhull = defect_vertices[hull.vertices,1]
        # xhull, yhull = self.__generate_spline(xhull, yhull, closed=True)

        defect_contour = np.asarray([defect_vertices[i] for i in hull.vertices])

        if not isinstance(defect_contour[:,0:2],Delaunay):
            hull = Delaunay(defect_contour[:,0:2])
        interior_defect = [pt for pt in np.asarray(self.skull.points)
            if hull.find_simplex(pt[0:2])>=0 and pt[2] > 0.20]

        # subtract those points from defect_vertices
        #   and remove remaining defect_vertices from skull.points
        defect_skull = open3d.geometry.PointCloud()
        defect_skull_vertices = [list(pt)
            for pt in np.asarray(self.skull.points)
            if pt not in np.asarray(interior_defect)]
        defect_skull.points = open3d.utility.Vector3dVector(np.asarray(defect_skull_vertices))

        # points on convex hull should be labelled as defect contour
        [x_max, y_max, z_max] = defect_skull.get_max_bound()
        [x_min, y_min, z_min] = defect_skull.get_min_bound()
        res = 15
        x_step = (x_max-x_min)/res
        y_step = (y_max-y_min)/res
        z_step = (z_max-z_min)/res
        
        input_data = np.zeros((res, res, res))
        # input_pcl = open3d.geometry.PointCloud()
        # input_pcl_vertices = []
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=np.asarray([x_min+i*x_step, y_min+j*y_step, z_min+k*z_step]).T,
                                                                max_bound=np.asarray([x_min+(i+1)*x_step, y_min+(j+1)*y_step, z_min+(k+1)*z_step]).T)
                    cropped = defect_skull.crop(bbox)
                    if cropped.has_points():
                        # input_pcl_vertices.append([x_min+(i+0.5)*x_step, y_min+(j+0.5)*y_step, z_min+(k+0.5)*z_step])
                        input_data[i,j,k] = 1
                        for l in np.asarray(cropped.points):
                            if l in np.asarray(defect_contour):
                                input_data[i,j,k] = 2

        # input_pcl.points = open3d.utility.Vector3dVector(np.asarray(input_pcl_vertices))
        open3d.visualization.draw_geometries([defect_skull, camera_frame])

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
        # surface = open3d.geometry.PointCloud()
        for i in range(len(camera)):
            pcd, _ = self.skull.hidden_point_removal(camera[i], radius)
            pts.extend(pcd.vertices)
        self.skull.points = open3d.utility.Vector3dVector(np.asarray(pts))
        trimesh = open3d.geometry.TriangleMesh()
        trimesh.vertices = self.skull.points
        trimesh = trimesh.remove_duplicated_vertices()
        self.skull.points = trimesh.vertices
        # open3d.visualization.draw_geometries([self.skull])

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

    def __generate_spline(self, x, y, closed=False, steps=20):
        """
        Credit: https://github.com/andrea-cuttone/geoplotlib/blob/master/geoplotlib/core.py

        catmullrom spline
        http://www.mvps.org/directx/articles/catmull/
        """

        if closed:
            x = x.tolist()
            x.insert(0, x[-1])
            x.append(x[1])
            x.append(x[2])

            y = y.tolist()
            y.insert(0, y[-1])
            y.append(y[1])
            y.append(y[2])

        points = np.vstack((x,y)).T

        curve = []

        if not closed:
            curve.append(points[0])

        for j in range(1, len(points)-2):
            for s in range(steps):
                t = 1. * s / steps
                p0, p1, p2, p3 = points[j-1], points[j], points[j+1], points[j+2]
                pnew = 0.5 *((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t**2 + (-p0 + 3*p1- 3*p2 + p3) * t**3)
                curve.append(pnew)

        if not closed:
            curve.append(points[-1])

        curve = np.array(curve)
        return curve[:, 0], curve[:, 1]


m = Mesh("./STL/CRANIAL HEADS_Head_1_001.stl")
m.remove_inner_layer()
m.extract_labeled_data()
