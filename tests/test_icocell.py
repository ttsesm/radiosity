# import numpy as np
# from utils import Isocell
#
# import matplotlib
# matplotlib.use('Qt5Agg')
#
# import pyvista as pv
# from mayavi import mlab
#
# def test_isocell():
#     isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)
#
#     # mlab.points3d(isocell.Xr, isocell.Yr, isocell.Zr, color=(1,0,0), scale_factor=.01)
#     # mlab.points3d(isocell.Xc, isocell.Yc, isocell.Zc, color=(0, 0, 1), scale_factor=.01)
#
#     points = np.column_stack([isocell.Xr, isocell.Yr, isocell.Zr])
#     cell_points = np.column_stack([isocell.Xc, isocell.Yc, isocell.Zc])
#     poly = pv.PolyData(points)
#     cell = pv.PolyData(cell_points)
#     # # poly["My Labels"] = ["P {}".format(i) for i in range(poly.n_points)]
#     #
#     plotter = pv.BackgroundPlotter()
#     # # plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)
#     plotter.add_mesh(poly, color="red", point_size=1)
#     plotter.add_mesh(cell, color="blue", point_size=1)
#
#
#     print('End testing the isocell module!!!!')
#
#
#
# if __name__ == '__main__':
#     print('Testing the Isocell module!!!!')
#     test_isocell()

import time

import pyvista
plotter = pyvista.BackgroundPlotter()
mesh = pyvista.Sphere()
plotter.add_mesh(mesh)
plotter.show()

# demonstrate non-blocking events
for i in range(100):
    mesh.points *= 1.01
    plotter.render()
    plotter.app.processEvents()

plotter.add_text('sleeping...')
time.sleep(3)  # demonstrate blocking event