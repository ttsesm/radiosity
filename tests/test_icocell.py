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
#     test = pv.BackgroundPlotter()
#     mesh = pv.Sphere()
#     test.add_mesh(mesh)
#
#
#     print('End testing the isocell module!!!!')
#     plotter.app.exec_()
#
#
#
# if __name__ == '__main__':
#     print('Testing the Isocell module!!!!')
#     test_isocell()
#     print('End!!!!')

#
# import time
#
# import pyvista
# plotter = pyvista.BackgroundPlotter()
# mesh = pyvista.Sphere()
# plotter.add_mesh(mesh)
# plotter.show()
#
# plotter.app.exec_()
#
# print('heloo')
# print("hello2")
#
# # # demonstrate non-blocking events
# # for i in range(100):
# #     mesh.points *= 1.01
# #     plotter.render()
# #     plotter.app.processEvents()
# #
# # plotter.add_text('sleeping...')
# # time.sleep(3)  # demonstrate blocking event


import time
import multiprocessing
import os
import sys
import pyvista

def plot_mesh():

    # from matplotlib.pyplot import plot, draw, show
    print("entered plotting process")
    # plot(data)
    # show() # this will block and remain a viable process as long as the figure window is open
    plotter = pyvista.BackgroundPlotter()
    # plotter = pyvista.Plotter()
    mesh = pyvista.Sphere()
    plotter.add_mesh(mesh)
    plotter.show()
    plotter.app.exec_()
    print("exiting plotting process")


if __name__ == "__main__":
    print("starting __main__")

    proc = multiprocessing.Process(target=plot_mesh, args=())
    proc.daemon = False
    proc.start()
    time.sleep(1)

    print("exiting main")
    os._exit(0) # this exits immediately with no cleanup or buffer flushing
    # sys.exit()  # this exits the main process