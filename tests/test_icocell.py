import numpy as np
import multiprocessing
from utils import Isocell

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
from mayavi import mlab


def test_isocell():
    isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)

    # # Plot points in 2D
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.scatter(isocell.Xr, isocell.Yr, color='r')
    # ax.scatter(isocell.Xc, isocell.Yc, color='b')
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # ax.set_title('Isocell Unit circle')
    # # plt.show()
    # plt.show(block=True)

    # # Plot points in 3D
    # mlab.points3d(isocell.Xr, isocell.Yr, isocell.Zr, color=(1,0,0), scale_factor=.01)
    # mlab.points3d(isocell.Xc, isocell.Yc, isocell.Zc, color=(0, 0, 1), scale_factor=.01)
    # mlab.show()

    points = np.column_stack([isocell.Xr, isocell.Yr, isocell.Zr])
    cell_points = np.column_stack([isocell.Xc, isocell.Yc, isocell.Zc])
    poly = pv.PolyData(points)
    cell = pv.PolyData(cell_points)
    # poly["My Labels"] = ["P {}".format(i) for i in range(poly.n_points)]

    plotter = pv.BackgroundPlotter()
    # plotter = pv.Plotter()
    # # plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)
    plotter.add_mesh(poly, color="red", point_size=1)
    plotter.add_mesh(cell, color="blue", point_size=1)
    plotter.add_axes()
    plotter.show()

    print('End testing the isocell module!!!!')
    # plt.show()



if __name__ == '__main__':
    print('Testing the Isocell module!!!!')
    test_isocell()