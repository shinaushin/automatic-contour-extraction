import numpy as np
import os

import open3d

def crop_geometry(pcd):
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
    open3d.visualization.draw_geometries_with_editing([pcd])

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


skull = open3d.io.read_triangle_mesh("./STL/CRANIAL HEADS_Head_1_001.stl")

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
downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

print(downpcd.get_center())
print(downpcd.get_max_bound())
print(downpcd.get_min_bound())

crop_geometry(downpcd)
