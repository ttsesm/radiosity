import numpy as np
import os
import copy
import multiprocessing
from utils import Isocell
from utils import LightDistributionCurve
from utils import FormFactor

from scipy.io import loadmat

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
import open3d as o3d
import trimesh
import pyembree
# from pyoctree import pyoctree as ot
import vtkplotter as vp

from numba import jit, cuda

def plot_mesh(verts, faces):
    # points = isocell.points
    # cell_points = np.column_stack([isocell.Xc, isocell.Yc, isocell.Zc])
    poly = pv.PolyData(verts, faces)
    # cell = pv.PolyData(cell_points)
    # # lines = pv.lines_from_points(cell.points)
    #
    plotter = pv.BackgroundPlotter()
    plotter.add_mesh(poly, show_edges=True)
    # # plotter.add_mesh(cell, color="blue", point_size=1)
    # # plotter.add_mesh(lines, color='blue')
    plotter.add_axes()
    plotter.show()
    plotter.app.exec_()


def test_formfactors():
    # load data from a .mat file
    data = loadmat('../cad_models/subdivided/cadModel_garchingVisionLab.mat')

    # manual retrieval (TODO: check if this can be handled automatically somehow)
    faces = data['cadModel_'][0][0][0] - 1
    # faces = np.insert(faces, 0, 3, axis=1)
    # vertices = np.vstack([[0,0,0], data['cadModel_'][0][0][1]])
    vertices = data['cadModel_'][0][0][1]
    floor_patches = data['cadModel_'][0][0][2] - 1
    ceiling_patches = data['cadModel_'][0][0][3] - 1
    wall_patches = data['cadModel_'][0][0][4][0][0][4] - 1
    light1_patches = data['cadModel_'][0][0][5][0][0][0] - 1
    light2_patches = data['cadModel_'][0][0][5][0][0][1] - 1
    light3_patches = data['cadModel_'][0][0][5][0][0][2] - 1
    light4_patches = data['cadModel_'][0][0][5][0][0][3] - 1
    light5_patches = data['cadModel_'][0][0][5][0][0][4] - 1
    light6_patches = data['cadModel_'][0][0][5][0][0][5] - 1
    light7_patches = data['cadModel_'][0][0][5][0][0][6] - 1
    light8_patches = data['cadModel_'][0][0][5][0][0][7] - 1
    light_patches = data['cadModel_'][0][0][5][0][0][8] - 1
    desk_patches = data['cadModel_'][0][0][6][0][0][8] - 1
    leg_desk_patches = data['cadModel_'][0][0][7] - 1
    panel_patches = data['cadModel_'][0][0][8] - 1
    panel_handle_patches = data['cadModel_'][0][0][9] - 1
    centroids = data['cadModel_'][0][0][10]
    normals = data['cadModel_'][0][0][11]
    areas = data['cadModel_'][0][0][12]

    # proc = multiprocessing.Process(target=plot_mesh, args=(vertices, faces[np.vstack([floor_patches, wall_patches, desk_patches, leg_desk_patches, panel_patches, panel_handle_patches, light_patches]),:].reshape(-1,4),))
    # proc.daemon = False
    # proc.start()
    # # faces = np.insert(faces, 0, 3, axis=1)
    # # # plot_mesh(vertices, faces)
    # # plot_mesh(vertices, faces[np.vstack([floor_patches, wall_patches, desk_patches, leg_desk_patches, panel_patches, panel_handle_patches, light_patches]),:].reshape(-1,4))

    # # open3d + point cloud example
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(vertices))
    #
    # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd])

    # open3d + mesh example
    # # mesh = o3d.geometry.TriangleMesh(vertices=np.asarray(vertices), faces=faces)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # # mesh.triangles = o3d.utility.Vector3iVector(faces[np.vstack([floor_patches, wall_patches, desk_patches, leg_desk_patches, panel_patches, panel_handle_patches, light_patches]),:].reshape(-1,3))
    # mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([mesh])

    # vector = np.array([0, 0, 1])
    # vector1 = np.array([0, 0, -1.3])
    # startpoint = np.array([0,0,-1.5])
    # m = trimesh.creation.icosphere()
    # # m.invert()
    #
    # # dots = np.dot(m.face_normals, vector)
    # # dots = dots / np.linalg.norm(startpoint - m.face_normals, axis=1)
    # # # # front_facing = dots < 1e-5
    # # front_facing = dots < 0
    #
    # directional_dist = startpoint - m.triangles_center
    # isPointingToward = np.einsum("ij,ij->i", directional_dist, m.face_normals) / np.linalg.norm(directional_dist, axis=1)
    #
    # # directional_dist = startpoint - m.triangles_center
    # # isInFOV = np.arccos(np.einsum("ij,ij->i", -1*directional_dist, m.face_normals) / np.linalg.norm(-1*directional_dist, axis=1))
    # isInFOV = np.arccos(np.dot(-1*directional_dist, vector) / np.linalg.norm(-1*directional_dist, axis=1))
    # front_facing = np.logical_and(isPointingToward > 0, isInFOV <= 90)
    # # front_facing = isPointingToward > 0
    #
    # # axes = vp.addons.buildAxes(vp.trimesh2vtk(m), c='k', zxGrid2=True)
    # ray = vp.Arrows(startpoint.reshape(-1,3), vector1.reshape(-1,3), c='g')
    # # vp.show(vp.trimesh2vtk(m).alpha(0.1).lw(0.1), normal, axes, axes=4)
    #
    # m.update_faces(front_facing)
    # axes = vp.addons.buildAxes(vp.trimesh2vtk(m), c='k', zxGrid2=True)
    # # normals = vp.Lines(m.triangles_center, m.triangles_center+m.face_normals, c='b', scale=.2)
    # # vp.show(normal, axes, axes=4)
    # vp.show(vp.trimesh2vtk(m).alpha(0.1).lw(0.1), normals, ray, axes, axes=4)

    # mesh = pv.PolyData(vertices, faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=normals, process=False, use_embree=True)
    # vp.show(vp.trimesh2vtk(mesh).alpha(0.1).lw(0.1), axes=4)

    # ff = FormFactor(mesh)
    # ffs = FormFactor(mesh).calculate_form_factors_matrix(processes=5)
    # ffs = FormFactor(mesh).calculate_form_factors_matrix(processes=multiprocessing.cpu_count() - 1)
    F = FormFactor(mesh)
    F.calculate_form_factors_matrix()
    F.apply_distribution_curve(patches=light_patches)

    print('End testing the form factors module!!!!')
    # plt.show()

# function optimized to run on gpu
@jit(target ='cuda')
def func2(a):
    for i in range(10000000):
        a[i]+= 1



if __name__ == '__main__':
    print('Testing the formfactors module!!!!')
    test_formfactors()
    # n = 10000000
    # a = np.ones(n, dtype=np.float64)
    # b = np.ones(n, dtype=np.float32)
    #
    # func2(a)
    os._exit(0)