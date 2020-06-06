import numpy as np
import multiprocessing
from utils import Isocell
from utils import rotation as r

from multiprocessing.dummy import Pool as ThreadPool

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
import vtkplotter as vp
from mayavi import mlab

# sphere = pv.Sphere(radius=0.85)
# start = [0, 0, 0]
# def __ray_casting(j, stop):
#     print(start)
#     print(stop)
#     points, ind = sphere.ray_trace(start, stop)
#
#     print(points)
#     print(ind)
#
#     return points, ind

def test_isocell_ray_casting():
    # Create source to ray trace
    sphere = pv.Sphere(radius=0.85)

    p1 = vp.Plotter()
    p2 = vp.Plotter()
    test_sphere = vp.Sphere()
    p1.show(test_sphere, axes=1)
    p2.show(test_sphere, axes=1)
    vp.show(1, pv.Cube(), axes=1)

    # faces = pv.convert_array(sphere.GetFaces())

    # Create isocell rays
    isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)

    drays = ((isocell.points @ r.vrrotvec2mat([0, 0, 1], [0, 0, 1]).T).T).reshape(-1,3)

    # Define line segment
    start = [0, 0, 0]
    # stop = [[0, 0, 1]]
    # stop = [0.25, 1, 0.5]
    # stop = ([0.25, 1, 0.5],[0.5, 1, 0.25])
    endPoints = drays

    # with ThreadPool(processes=4) as pool:
    #     points1, ind1 = zip(*pool.starmap(__ray_casting, enumerate(stop)))

    # Render the result
    p = pv.Plotter()
    p.add_mesh(sphere,
               show_edges=True, opacity=0.5, color="w",
               lighting=False, label="Test Mesh")

    for i, stop in enumerate(drays):

        # Perform ray trace
        points, ind = sphere.ray_trace(start, stop)

        # Create geometry to represent ray trace
        ray = pv.Line(start, stop)
        intersection = pv.PolyData(points)



        # pv.plot_arrows(np.array(start),np.array(stop))
        # pv.plot_arrows(np.array(start),np.array(stop))
        p.add_mesh(ray, color="blue", line_width=5, label="Ray Segment")
        p.add_mesh(intersection, color="maroon",
                   point_size=25, label="Intersection Points")
    p.add_legend()
    p.show()


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
    plotter.app.exec_()

    print('End testing the isocell module!!!!')
    # plt.show()


if __name__ == '__main__':
    print('Testing the Isocell module!!!!')
    # test_isocell()
    test_isocell_ray_casting()


# import trimesh
# import pyvista as pv
#
# mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
#                        faces=[[0, 1, 2]],
#                        process=False)
#
#
# print(pv.Report())