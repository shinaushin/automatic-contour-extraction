import matplotlib
matplotlib.use('tkagg')

from matplotlib import pyplot
from mpl_toolkits import mplot3d
import numpy

import open3d

skull = open3d.io.read_triangle_mesh('./STL/CRANIAL HEADS_Head_1_001.stl')

figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

pcd = open3d.geometry.PointCloud()
pcd.points = skull.vertices
pcd.colors = skull.vertex_colors
pcd.normals = skull.vertex_normals
open3d.visualization.draw_geometries([pcd])
