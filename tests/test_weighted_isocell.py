import numpy as np
import multiprocessing
from utils import Isocell
from utils import LightDistributionCurve

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
from mayavi import mlab


def test_isocell():
    isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)
    ldc = LightDistributionCurve()

    isocell.compute_weights(ldc.properties['symmetric_ldc'], np.array([[0, 0, 0], [1,1,1]]))


    points = isocell.points
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