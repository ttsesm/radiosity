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
    isocell = Isocell(rays=2000, div=3, isrand=0, draw_cells=True)
    ldc = LightDistributionCurve()

    isocell.compute_weights(ldc.properties['symmetric_ldc'], np.array([[0, 0, 0]]))


    points = isocell.points
    cell_points = np.column_stack([isocell.Xc, isocell.Yc, isocell.Zc])
    poly = pv.PolyData(points)
    cell = pv.PolyData(cell_points)
    # lines = pv.lines_from_points(cell.points)

    plotter = pv.BackgroundPlotter()
    plotter.add_mesh(poly, scalars=isocell.weights, point_size=2)
    # plotter.add_mesh(cell, color="blue", point_size=1)
    # plotter.add_mesh(lines, color='blue')
    plotter.add_axes()
    plotter.show()
    plotter.app.exec_()

    print('End testing the isocell module!!!!')
    # plt.show()



if __name__ == '__main__':
    print('Testing the Isocell module!!!!')
    test_isocell()