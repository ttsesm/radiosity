import numpy as np
import os
import time
import multiprocessing
from utils import LightDistributionCurve

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
from mayavi import mlab


def test_ldc():
    ldc = LightDistributionCurve()
    # ldc.plot2D(inline=False, type='normalized')
    ldc.plot3D()

    # curve = np.array([[1.,       1.,      1.,     1.,      1.,      1.,      1.,      1.,      1.,      1.],
    #        [0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984],
    #        [0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936],
    #        [0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854],
    #        [0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736],
    #        [0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572],
    #        [0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313],
    #        [0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770],
    #        [0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
    #        [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00]])
    #
    # lsc = LightDistributionCurve(curve)
    # lsc.plot2D(color='blue', legend='LSC')
    # # plt.show()

    print('End testing the LightDistributionCurve module!!!!')

def plot_mesh(data):
    print("entered plotting process")
    plotter = pv.BackgroundPlotter()
    mesh = pv.Sphere()
    plotter.add_mesh(mesh)
    plotter.show()
    plotter.app.exec_()
    print("exiting plotting process")



if __name__ == '__main__':
    print('Testing the LightDistributionCurve module!!!!')
    proc = multiprocessing.Process(target=plot_mesh, args=([],))
    proc.daemon = False
    proc.start()
    time.sleep(1)
    # test_ldc()
    print("exiting main")
    os._exit(0)  # this exits immediately with no cleanup or buffer flushing