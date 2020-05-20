import numpy as np
from utils import Isocell

import matplotlib
matplotlib.use('Qt5Agg')

import pyvista as pv
from mayavi import mlab

def test_isocell():
    isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)

    # mlab.points3d(isocell.Xr, isocell.Yr, isocell.Zr, color=(1,0,0), scale_factor=.01)
    # mlab.points3d(isocell.Xc, isocell.Yc, isocell.Zc, color=(0, 0, 1), scale_factor=.01)

    points = np.column_stack([isocell.Xr, isocell.Yr, isocell.Zr])
    cell_points = np.column_stack([isocell.Xc, isocell.Yc, isocell.Zc])
    poly = pv.PolyData(points)
    cell = pv.PolyData(cell_points)
    # # poly["My Labels"] = ["P {}".format(i) for i in range(poly.n_points)]
    #
    plotter = pv.BackgroundPlotter()
    # # plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)
    plotter.add_mesh(poly, color="red", point_size=1)
    plotter.add_mesh(cell, color="blue", point_size=1)


    print('End testing the isocell module!!!!')



if __name__ == '__main__':
    print('Testing the Isocell module!!!!')
    test_isocell()