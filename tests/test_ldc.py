import time
import multiprocessing
import os
import sys
import pyvista as pv

import numpy as np

from utils import LightDistributionCurve
from utils import rotation as r

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

# def plotting_thread():
#     ldc = LightDistributionCurve()
#     ldc.plot2D(inline=True, type='normalized')
#
# def plotting_thread2():
#     ldc = LightDistributionCurve()
#     ldc.plot3D()

def test_vvrotvec():
    rot = r.vvrotvec([0, 0, 0], [0, 1, 0])

    M = r.vrrotvec2mat(rot)

    print(rot)


def test_ldc():

    ldc = LightDistributionCurve()
    ldc.plot2D(holdOn=False, type='normalized')
    ldc.plot3D()

    curve = np.array([[1.,       1.,      1.,     1.,      1.,      1.,      1.,      1.,      1.,      1.],
           [0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984,   0.984],
           [0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936,   0.936],
           [0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854,   0.854],
           [0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736,   0.736],
           [0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572,   0.572],
           [0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313,   0.313],
           [0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770,  0.0770],
           [0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164,  0.0164],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00],
           [0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00,    0.00]])

    lsc = LightDistributionCurve(curve)
    lsc.plot2D(color='blue', legend='LSC')
    # plt.show()

    print('End testing the LightDistributionCurve module!!!!')

# def plot_mesh():
#     print("entered plotting process")
#     plotter = pv.BackgroundPlotter()
#     mesh = pv.Sphere()
#     plotter.add_mesh(mesh)
#     plotter.show()
#     plotter.app.exec_()
#     print("exiting plotting process")

if __name__ == '__main__':
    print('Testing the LightDistributionCurve module!!!!')
    # proc = multiprocessing.Process(target=plotting_thread, args=())
    # proc.daemon = False
    # proc.start()
    # proc2 = multiprocessing.Process(target=plotting_thread2, args=())
    # proc2.daemon = False
    # proc2.start()
    # # time.sleep(1)

    test_vvrotvec()
    test_ldc()
    print("exiting main")
    os._exit(0)  # this exits immediately with no cleanup or buffer flushing