# import numpy as np
# from utils import Isocell
#
import matplotlib
import matplotlib.pyplot as plt
# plt.ion()
matplotlib.use('Qt5Agg')
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
# import multiprocessing
import pathos.multiprocessing
import os
import sys
import pyvista
import numpy as np
import vtkplotter as vp
import psutil
import vtk

def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;

def plot_mesh_vtkplotter():
    print(pathos.multiprocessing.current_process())
    print(os.getpid())
    sphere = vp.Sphere().smoothWSinc().x(0.2).alpha(0.1).lw(0.1)
    cube = vp.Cube()

    p1 = vp.Plotter()

    p1.show(sphere, cube, interactive=True)

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

def plot_graph(data, color, holdOn):

    print("entered plotting process")
    ax = plt.subplot(111, projection='polar')
    ax.plot(data[0], data[1], color=color)
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')

    if not holdOn:
        print("Not on hold!!!")
        plt.show() # this will block and remain a viable process as long as the figure window is open
    print("exiting plotting process")


if __name__ == "__main__":
    print("starting __main__")

    # fig = plt.figure()
    # fig = plt.subplot(111, projection='polar')
    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r
    # fig.plot(theta, r, color='red')
    # proc = multiprocessing.Process(target=plot_mesh, args=())
    # proc = multiprocessing.Process(target=plot_graph, args=([theta, r], 'red', False,))
    proc = pathos.helpers.mp.process.Process(target=plot_mesh_vtkplotter, args=())
    
    proc.daemon = False
    proc.start()
    time.sleep(1)

    # Check if any chrome process was running or not.
    if checkIfProcessRunning('Python'):
        print('Yes a vtkplotter process was running')
    else:
        print('No vtkplotter process was running')

    print(vp.settings.plotter_instance)  # this prints None

    # vp.show(0,vp.Cube())

    # plt.show()

    print("exiting main")
    os._exit(0) # this exits immediately with no cleanup or buffer flushing
#     # sys.exit()  # this exits the main process