import numpy as np
import os
import copy
import multiprocessing
from utils import Isocell
from utils import LightDistributionCurve

from scipy.io import loadmat

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
import open3d as o3d

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
    faces = np.insert(faces, 0, 3, axis=1)
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(vertices))

    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])

    print('End testing the form factors module!!!!')
    # plt.show()



if __name__ == '__main__':
    print('Testing the formfactors module!!!!')
    test_formfactors()
    os._exit(0)