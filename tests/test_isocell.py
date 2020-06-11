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
# vp.settings.useDepthPeeling=True
# vp.settings.useFXAA=True

import trimesh

def test_isocell_ray_casting():
    # Create source to ray trace
    # sphere = pv.Sphere(radius=0.85)
    sphere1 = vp.Sphere(r=0.85, c="r", alpha=0.1).lw(0.1)
    mesh = trimesh.Trimesh(vertices=sphere1.points(), faces=sphere1.faces(), process=False, use_embree=True)
    # mesh = trimesh.creation.icosphere()


    # Create isocell rays
    isocell = Isocell(rays=1000, div=5, isrand=0, draw_cells=True)

    drays = ((isocell.points @ r.vrrotvec2mat([0, 0, 1], [0, 0, 1]).T).T).reshape(-1,3)

    # Define line segment
    start = [0, 0, 0]
    endPoints = drays

    # intersects_location requires origins to be the same shape as vectors
    origins = np.tile(np.expand_dims(start, 0), (len(drays), 1))

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(origins, drays, multiple_hits=False)

    # for each hit, find the distance along its vector
    # you could also do this against the single camera Z vector
    depth = trimesh.util.diagonal_dot(points - start, drays[index_ray])

    locs = trimesh.points.PointCloud(points)

    # # stack rays into line segments for visualization as Path3D
    # ray_visualize = trimesh.load_path(np.hstack((origins, origins + drays * 1.2)).reshape(-1, 2, 3))

    # render the result with vtkplotter
    axes = vp.addons.buildAxes(vp.trimesh2vtk(mesh), c='k', zxGrid2=True)
    # vp.show(mesh, ray_visualize, axes, locs, axes=4)
    lines = vp.Lines(origins,drays, c='b')
    vp.show(vp.trimesh2vtk(mesh).alpha(0.1).lw(0.1), lines, axes, locs, axes=4)

    # Render the result with pyvista
    sphere = pv.PolyData(mesh.vertices, np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)))
    p = pv.Plotter()
    p.add_mesh(sphere,
               show_edges=True, opacity=0.5, color="w",
               lighting=False, label="Test Mesh")
    # p.add_arrows(origins[1,:], drays[1,:], mag=1, color=True, opacity=0.5, )
    p.add_lines(np.hstack([origins,drays]).reshape(-1,3), color='b')
    # intersections = pv.PolyData(points)
    # p.add_mesh(intersections, color="maroon", point_size=25, label="Intersection Points")

    # for i, stop in enumerate(drays):
    #
    #     # Perform ray trace
    #     points, ind = sphere.ray_trace(start, stop)
    #
    #     # Create geometry to represent ray trace
    #     ray = pv.Line(start, stop)
    #     intersection = pv.PolyData(points)
    #
    #
    #
    #     # pv.plot_arrows(np.array(start),np.array(stop))
    #     # pv.plot_arrows(np.array(start),np.array(stop))
    #     p.add_mesh(ray, color="blue", line_width=5, label="Ray Segment")
    #     p.add_mesh(intersection, color="maroon",
    #                point_size=25, label="Intersection Points")
    # p.add_legend()
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