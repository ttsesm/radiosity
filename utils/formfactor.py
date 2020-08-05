#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
# from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import numpy_indexed as npi
# from numpy.core.umath_tests import inner1d

from scipy import interpolate

import time
import open3d as o3d

import pyvista as pv

from utils.triangle import Triangle, vectorize, distance

from utils import Isocell
from utils import LightDistributionCurve
import trimesh
import pyembree
import vtkplotter as vp

from utils import rotation as r

from joblib import Parallel, delayed

from numba import jit, cuda
import matplotlib.pylab as plt

import copy


class FormFactor(object):
    def __init__(self, *args, **kwargs):

        if not args:
            return
        elif len(args) == 1:
            # if isinstance(args[0], pv.PolyData):
            if isinstance(args[0], trimesh.Trimesh):
                self.mesh = args[0]
            else:
                raise TypeError('Invalid input type')

        # self.f # scene faces
        # self.v # scene vertices

        self.__patch_count = self.mesh.faces.shape[0]
        # self.__patch_count = self.mesh.n_faces
        # self.__patch_count = 100

        # self.ffs = np.zeros(shape=(self.mesh.faces.size, self.mesh.faces.size),dtype='float32')
        # self.ffs = np.ones(shape=(self.mesh.faces.size, self.mesh.faces.size),dtype='float32')
        # self.ffs = np.ones(shape=(self.__patch_count, self.__patch_count), dtype='float32')
        self.ffs = np.zeros(shape=(self.__patch_count, self.__patch_count), dtype='float32')
        # self.ffs = np.ones(shape=(100,100),dtype='float32')

        self.__n_rays = kwargs.pop('rays', 1000)
        self.isocell = Isocell(rays=self.__n_rays, div=3, isrand=0, draw_cells=True)

        # correct the number of rays created from the isocell casting
        self.__n_rays = self.isocell.points.shape[0]
        self.__form_factor_properties = {}

    # # function optimized to run on gpu
    # @jit(target="cuda")
    def __calculate_form_factors(self):

        # calculate the rotation matrix between the face normal and the isocell unit sphere's
        # original position and rotate the rays accordingly to match the face normal direction
        face_normals = self.mesh.face_normals
        # # test1 = r.vvrotvec(self.mesh.face_normals[1,:], [0, 0, 1])
        # test1 = r.vvrotvec(face_normals[591:593,:], [0, 0, 1])
        # test = r.vvrotvec(face_normals, [0, 0, 1])
        # rotation_matrices = r.vrrotvec2mat(face_normals, [0, 0, 1])
        rotation_matrices = r.rotation_matrices_from_vectors(face_normals, [0, 0, 1])
        self.__form_factor_properties['rotation_matrices'] = rotation_matrices

        # drays = np.einsum('ijk,ak->iak', rotation_matrices, self.isocell.points)
        # drays = np.einsum('ijj,aj->iaj', rotation_matrices, self.isocell.points)
        drays = np.einsum('ijk,aj->iak', rotation_matrices, self.isocell.points)
        self.__form_factor_properties['drays'] = drays

        # get the centroid of the face/patch and shift it a bit so that rays do not stop at the self face thrown from
        start_points = self.mesh.triangles_center  # the face/patch center points
        offset = np.sign(face_normals)
        offset = offset * 1e-3
        origins = start_points + offset

        # intersects_location requires origins to be the same shape as vectors
        origins = np.repeat(origins, self.__n_rays, axis=0)
        self.__form_factor_properties['origins'] = origins.reshape(drays.shape[0], drays.shape[1], -1)
        # origins = np.tile(np.expand_dims(start_points, 0), (drays.shape[0], 1)) + offset

        # tree = ot.PyOctree(self.mesh.vertices.copy(order='C'), self.mesh.faces.copy(order='C').astype(np.int32))
        # rayList = np.array([origins, drays.reshape(-1, 3)], dtype=np.float32)
        # startPoint = [0.0, 0.0, 0.0]
        # endPoint = [0.0, 0.0, 1.0]
        # rayList1 = np.array([[startPoint, endPoint]], dtype=np.float32)
        # intersectionFound = tree.rayIntersection(rayList)

        # start casting and find intersection points, rays and faces
        start = time.time()
        intersection_points, index_ray, index_tri = self.mesh.ray.intersects_location(origins, drays.reshape(-1, 3), multiple_hits=False)
        end = time.time()
        print('Ray casting in: {} sec'.format(end - start))
        # tree = ot.PyOctree(vertices.copy(order='C'), faces.copy(order='C').astype(np.int32))

        # check whether there were / print intersection points
        print('Intersections: {}/{}'.format(len(intersection_points), len(origins)))

        # check whether the extracted intersection output size is correct and fits the input
        if intersection_points.shape[0]!=index_ray.shape[0]!=index_tri.shape[0]:
            raise Exception('bad size alignment to the intersection ouput matrices')

        # find the indices of rays that did not intersect to any face and recover the size of the total casted rays
        no_intersection_rays = np.arange(origins.shape[0])
        idxs_of_no_intersection_rays = no_intersection_rays[~np.isin(np.arange(no_intersection_rays.size), index_ray)]

        # check whether there are no_intersection_rays, and if yes adjust sizes in the output
        if idxs_of_no_intersection_rays.any():
            # first apply backface culling and filter intersections from rays hitting faces from the back side
            # TODO: this could be addressed optimally by embree if it gets compiled with the corresponding parameter
            start = time.time()
            front_facing = self.__isFacing(np.delete(origins, idxs_of_no_intersection_rays, axis=0), np.delete(np.repeat(face_normals, self.__n_rays, axis=0), idxs_of_no_intersection_rays, axis=0), index_tri)
            end = time.time()
            print('Backface pulling in: {} sec'.format(end - start))

            index_ray[np.where(front_facing == False)] = -1
            index_tri[np.where(front_facing == False)] = -1
            intersection_points[np.where(front_facing == False)] = -np.inf

            # index_ray = np.insert(index_ray, idxs_of_no_intersection_rays, -1) # simple insert does not work properly. See: https://stackoverflow.com/questions/47442115/insert-values-at-specific-locations-in-numpy-array-np-insert-done-right
            index_ray = np.insert(index_ray, idxs_of_no_intersection_rays - np.arange(len(idxs_of_no_intersection_rays)), -1)
            # index_tri = np.insert(index_tri, idxs_of_no_intersection_rays, -1)
            index_tri = np.insert(index_tri, idxs_of_no_intersection_rays - np.arange(len(idxs_of_no_intersection_rays)), -1)
            # intersection_points = np.insert(intersection_points, idxs_of_no_intersection_rays, -np.inf, axis=0)
            intersection_points = np.insert(intersection_points, idxs_of_no_intersection_rays - np.arange(len(idxs_of_no_intersection_rays)), -np.inf, axis=0)

        else:
            front_facing = self.__isFacing(origins, np.repeat(face_normals, self.__n_rays, axis=0), index_tri)
            index_ray[np.where(front_facing == False)] = -1
            index_tri[np.where(front_facing == False)] = -1
            intersection_points[np.where(front_facing == False)] = -np.inf

        index_ray = index_ray.reshape(self.__patch_count, -1)
        index_tri = index_tri.reshape(self.__patch_count, -1)
        intersection_points = intersection_points.reshape(self.__patch_count, self.__n_rays, -1)

        self.__form_factor_properties['index_rays'] = index_ray
        self.__form_factor_properties['index_tri'] = index_tri
        self.__form_factor_properties['intersection_points'] = intersection_points

        # Bin elements per row (this means to find how many times each face is intersected from the thrown rays) from the intersected triangles matrix, i.e. index_tri.
        # See:
        # https://stackoverflow.com/questions/62662346/map-amount-of-repeated-elements-row-wise-from-a-numpy-array-to-another?noredirect=1#comment110814113_62662346
        # https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy and
        # https://stackoverflow.com/a/40593110/1476932
        # solution addapted from the last link
        rowidx, colidx = np.indices(index_tri.shape)
        (cols, rows), B = npi.count((index_tri.flatten(), rowidx.flatten()))

        # remove negative indexing that we introduced from the missing intersections
        negative_idxs = np.where(cols < 0)
        cols = np.delete(cols, negative_idxs)
        rows = np.delete(rows, negative_idxs)
        B = np.delete(B, negative_idxs)

        # assign values to the corresponding position of the form factors matrix
        self.ffs[rows, cols] = B
        self.ffs /= self.__n_rays

        # # # check whether there are no_intersection_rays, and if yes adjust sizes in the output
        # # if idxs_of_no_intersection_rays.any():
        # #     # first apply backface culling and filter intersections from rays hitting faces from the back side
        # #     # TODO: this could be addressed optimally by embree if it gets compiled with the corresponding parameter
        # #     front_facing = self.__isFacing(np.delete(origins, idxs_of_no_intersection_rays, axis=0), np.delete(np.repeat(face_normals, self.isocell.points.shape[0], axis=0), idxs_of_no_intersection_rays, axis=0), index_tri)
        # #
        # #     index_ray = np.delete(index_ray, np.where(front_facing == False))
        # #     index_tri = np.delete(index_tri, np.where(front_facing == False))
        # #     intersection_points = np.delete(intersection_points, np.where(front_facing == False), axis=0)
        # #
        # # eq = npi.group_by(origins[index_ray])
        #
        # # locs = trimesh.points.PointCloud(intersection_points)
        #
        # # render the result with vtkplotter
        # axes = vp.addons.buildAxes(vp.trimesh2vtk(self.mesh), c='k', zxGrid2=True)
        # rays = vp.Lines(origins[0:1083, :], drays[0, 0:1083, :].reshape(-1, 3)+origins[0:1083, :], c='b', scale=200)
        # locs = vp.Points(intersection_points[0:1083, :], c='r')
        # # rays = vp.Arrows(origins, drays+start_point, c='b', scale=1000)
        # normal = vp.Arrows(start_points[0, :].reshape(-1, 3), (face_normals[0, :]+start_points[0, :]).reshape(-1, 3), c='g', scale=250)
        # vp.show(vp.trimesh2vtk(self.mesh).alpha(0.1).lw(0.1), locs, rays, normal, axes, axes=4)
        #
        # # # for each hit, find the distance along its vector
        # # # you could also do this against the single camera Z vector
        # # depth = trimesh.util.diagonal_dot(intersection_points - start_point, drays[index_ray])

        return self.ffs

    def __isFacing(self, startpoints, startpoints_normals, index_tri):

        directional_dist = startpoints - self.mesh.triangles_center[index_tri]
        isPointingToward = np.einsum("ij,ij->i", directional_dist, self.mesh.face_normals[index_tri]) / np.linalg.norm(directional_dist, axis=1)

        # directional_dist = startpoint - m.triangles_center
        isInFOV = np.arccos(np.einsum("ij,ij->i", -1*directional_dist, startpoints_normals) / np.linalg.norm(-1*directional_dist, axis=1))
        # isInFOV = np.arccos(np.dot(-1 * directional_dist, vector) / np.linalg.norm(-1 * directional_dist, axis=1))
        front_facing = np.logical_and(isPointingToward > 0, isInFOV <= 90)

        # # Plot example of backface intersections
        # idxs = np.where(front_facing == False)
        # axes = vp.addons.buildAxes(vp.trimesh2vtk(self.mesh), c='k', zxGrid2=True)
        #
        # normal = vp.Arrows(startpoints[idxs[0][10000], :].reshape(-1, 3), (startpoints_normals[idxs[0][10000], :] + startpoints[idxs[0][10000], :]).reshape(-1, 3), c='g', scale=250)
        # normal1 = vp.Arrows(self.mesh.triangles_center[index_tri[idxs[0][10000]], :].reshape(-1, 3), (self.mesh.face_normals[index_tri[idxs[0][10000]], :] + self.mesh.triangles_center[index_tri[idxs[0][10000]], :]).reshape(-1, 3), c='b', scale=250)
        #
        # # normal = vp.Arrows(startpoints[idxs[0], :].reshape(-1, 3), (startpoints_normals[idxs[0], :] + startpoints[idxs[0], :]).reshape(-1, 3), c='g', scale=250)
        # # normal1 = vp.Arrows(self.mesh.triangles_center[index_tri[idxs[0]], :].reshape(-1, 3), (self.mesh.face_normals[index_tri[idxs[0]], :] + self.mesh.triangles_center[index_tri[idxs[0]], :]).reshape(-1, 3), c='b', scale=250)
        #
        #
        # vp.show(vp.trimesh2vtk(self.mesh).alpha(0.1).lw(0.1), normal, normal1, axes, axes=4)

        return front_facing

    def __calculate_one_patch_form_factor(self, i, p_1):
        # print('[form factor] patch {}/{} ...'.format(i, self.__patch_count))
        # create empty form factor array
        # ff = np.ones(self.__patch_count)*i
        self.ffs[i,:] *= i

        # calculate the rotation matrix between the face normal and the isocell unit sphere's
        # original position and rotate the rays accordingly to match the face normal direction
        face_normal = self.mesh.face_normals[i,:].reshape(-1,3) # corresponding face normal
        drays = ((self.isocell.points @ r.vrrotvec2mat(face_normal, [0, 0, 1]).T).T).reshape(-1, 3)

        # get the centroid of the face/patch and shift it a bit so that rays do not stop at the self face thrown from
        start_point = self.mesh.triangles_center[i, :]  # the face/patch center point
        offset = np.sign(face_normal)
        offset = offset * 1e-5

        # intersects_location requires origins to be the same shape as vectors
        origins = np.tile(np.expand_dims(start_point, 0), (drays.shape[0], 1)) + offset
        # lines = vp.Lines(origins,drays+start_point, c='b', scale=1100)
        # vp.show(lines, axes=1)
        # origins[:, 0] = origins[:, 0] - 100

        # do the actual ray- mesh queries
        intersection_points, index_ray, index_tri = self.mesh.ray.intersects_location(origins, drays, multiple_hits=False)

        # locs = trimesh.points.PointCloud(intersection_points)

        # # render the result with vtkplotter
        # axes = vp.addons.buildAxes(vp.trimesh2vtk(self.mesh), c='k', zxGrid2=True)
        # lines = vp.Lines(origins, drays+origins, c='b', scale=200)
        # locs = vp.Points(intersection_points, c='r')
        # # rays = vp.Arrows(origins, drays+start_point, c='b', scale=1000)
        # normal = vp.Arrows(start_point.reshape(-1,3), face_normal+start_point.reshape(-1,3), c='g', scale=250)
        # vp.show(vp.trimesh2vtk(self.mesh).alpha(0.1).lw(0.1), locs, lines, normal, axes, axes=4)
        #
        # # for each hit, find the distance along its vector
        # # you could also do this against the single camera Z vector
        # depth = trimesh.util.diagonal_dot(intersection_points - start_point, drays[index_ray])

        # return ff
        print('[form factor] patch {}/{} ... intersections: {}'.format(i, self.__patch_count, len(intersection_points)))

    def __get_faces(self, faces, num_of_vertices=3):
        return faces.reshape(-1, num_of_vertices+1)[:, 1:num_of_vertices+1]

    def apply_distribution_curve(self, curve=np.array([]), patches=np.array([]), type='ldc'):

        if not patches.any():
            raise Exception('You need to provide patches where the distribution to be applied.')

        # create distribution curve
        distribution = LightDistributionCurve(curve)

        # calculate weights given the above distribution and the corresponding patches (specify the weight of each corresponding emmited ray)
        # TODO: this possibly could be solved within the isocell.compute_weights in a vectorized way somehow
        weights = Parallel(n_jobs=5, prefer="threads")(delayed(self.__compute_weights)(distribution.properties['symmetric_ldc'], i, p_1) for i, p_1 in enumerate(patches))
        # normalize the corresponding weights
        normalized_weights = self.__n_rays * np.stack(weights).squeeze() / np.sum(weights, axis=-1)

        # apply the corresponding binning based on the weights computed above.
        # see also:
        # https://stackoverflow.com/questions/62719951/weighted-numpy-bincount-for-2d-ids-array-and-1d-weights
        self.ffs[patches.squeeze(), :] = self.__bincount2D(self.__form_factor_properties['index_tri'][patches.squeeze(), :], normalized_weights, sz=self.ffs[patches.squeeze(), :].shape) / self.__n_rays


    def __bincount2D(self, id_ar_2D, weights, sz=None):
        # Inputs : 2D id array, 1D weights array

        # Extent of bins per col
        if sz == None:
            n = id_ar_2D.max() + 1
            N = len(id_ar_2D)
        else:
            n = sz[1]
            N = sz[0]

        if id_ar_2D.shape == weights.shape:
            W = weights.ravel()
        else:
            W = np.tile(weights, N)

        # add offsets to the original values to be used when we apply raveling later on
        id_ar_2D_offsetted = id_ar_2D + n * np.arange(N)[:, None]

        # Finally use bincount with those 2D bins as flattened and with
        # flattened b as weights. Reshaping is needed to add back into "a".
        ids = id_ar_2D_offsetted.ravel()

        return np.bincount(ids, W, minlength=n * N).reshape(-1, n)

        # self.isocell.compute_weights(distribution.properties['symmetric_ldc'], np.array([[0, 0, 0]]), type)

    # def apply_ldc(self, curve=np.array([]), patches=np.array([])):
    #
    #     if not patches.any():
    #         raise Exception('You need to provide patches where the distribution to be applied.')
    #
    #     self.__apply_distribution_curve(curve, patches)
    #
    #
    # def appl_lsc(self, curve=np.array([]), patches=np.array([])):
    #
    #     if not patches.any():
    #         raise Exception('You need to provide patches where the distribution to be applied.')
    #
    #     self.__apply_distribution_curve(curve, patches)

    def __compute_weights(self, distribution, i, patch):
        ''' Compute weight for the isocell rays in correspondence to given origin points, which could be the center of patches for example.'''
        rayPnts = self.__form_factor_properties['drays'][patch, :].reshape(-1, 3) + self.__form_factor_properties['origins'][patch,:].reshape(-1, 3)
        isocell_points = self.__form_factor_properties['intersection_points'][patch].reshape(-1, 3)

        if np.where(np.all(np.isinf(isocell_points), axis=1)):
            isocell_points[np.where(np.all(np.isinf(isocell_points), axis=1))] = rayPnts[np.where(np.all(np.isinf(isocell_points), axis=1))]

        n_vis = isocell_points.shape[0]
        n_pis = self.mesh.triangles_center[patch].shape[0]

        anglesX = np.linspace(0, 180, distribution.shape[0])
        anglesZ = np.linspace(0, 180, distribution.shape[1])

        # Compute the relative position of each point in the isocell sphere with respect to the given point
        v_points = (isocell_points - np.repeat(self.mesh.triangles_center[patch][np.newaxis, :], n_vis, axis=1)).reshape(-1,3)

        # Compute the distance between each point in the isocell sphere with respect to the given point
        norms = np.sqrt(np.sum(v_points * v_points, axis=1))

        # Compute the angle between the principal axis (Z-axis) and the ray connecting the light source to each point
        v_centerZ = self.__form_factor_properties['rotation_matrices'][patch, 2, :].flatten()
        vcenter_to_vpoint_angle_Z_axis = np.rad2deg(np.arccos(np.sum(np.repeat(v_centerZ[np.newaxis, :], n_vis*n_pis, axis=0) * v_points, axis=1) / norms))

        # Compute the angle between the principal axis (X-axis) and the ray connecting the light source to each point
        v_centerX = self.__form_factor_properties['rotation_matrices'][patch, 0, :].flatten()
        vcenter_to_vpoint_angle_X_axis = np.rad2deg(np.arccos(np.sum(np.repeat(v_centerX[np.newaxis, :], n_vis*n_pis, axis=0) * v_points, axis=1) / norms))

        interpolation = interpolate.interp2d(anglesX, anglesZ, distribution.T)

        weights = np.diagonal(interpolation(vcenter_to_vpoint_angle_Z_axis, vcenter_to_vpoint_angle_X_axis)).reshape(-1,n_vis)

        # self.weights[type] = np.array(weights)
        return  weights

    # def compute_weights(self, distribution, isocell_points, points, v_centerZ = np.array([0, 0, 1]), v_centerX = np.array([1, 0, 0]), type='ldc'):
    #     ''' Compute weight for the isocell rays in correspondence to given origin points, which could be the center of patches for example.'''
    #     n_vis = self.__n_rays
    #     n_pis = points.shape[0]
    #
    #     anglesX = np.linspace(0, 180, distribution.shape[0])
    #     anglesZ = np.linspace(0, 180, distribution.shape[1])
    #
    #     # Compute the relative position of each point in the isocell sphere with respect to the given point
    #     v_points = (isocell_points - np.repeat(points[np.newaxis, :], n_vis, axis=1)).reshape(-1,3)
    #
    #     # Compute the distance between each point in the isocell sphere with respect to the given point
    #     norms = np.sqrt(np.sum(v_points * v_points, axis=1))
    #
    #     # Compute the angle between the principal axis (Z-axis) and the ray connecting the light source to each point
    #     # v_centerZ = np.array([0, 0, 1])
    #     vcenter_to_vpoint_angle_Z_axis = np.rad2deg(np.arccos(np.sum(np.repeat(v_centerZ[np.newaxis, :], n_vis*n_pis, axis=0) * v_points, axis=1) / norms))
    #
    #     # Compute the angle between the principal axis (X-axis) and the ray connecting the light source to each point
    #     # v_centerX = np.array([1, 0, 0])
    #     vcenter_to_vpoint_angle_X_axis = np.rad2deg(np.arccos(np.sum(np.repeat(v_centerX[np.newaxis, :], n_vis*n_pis, axis=0) * v_points, axis=1) / norms))
    #
    #     interpolation = interpolate.interp2d(anglesX, anglesZ, distribution.T)
    #
    #     weights = np.diagonal(interpolation(vcenter_to_vpoint_angle_Z_axis, vcenter_to_vpoint_angle_X_axis)).reshape(-1,n_vis)
    #
    #     # self.weights[type] = np.array(weights)
    #     return  weights


    def calculate_form_factors_matrix(self, processes=4, **kwargs):

        ffs = self.__calculate_form_factors()

        # # Parallel(n_jobs=5, prefer="threads")(delayed(self.__calculate_one_patch_form_factor)(i, p_1) for i, p_1 in enumerate(self.mesh.faces))
        #
        # for i, p_i in enumerate(self.mesh.faces):
        #     self.__calculate_one_patch_form_factor(i, p_i)
        #
        # with ThreadPool(processes=processes) as pool:
        #     # for i, num in enumerate(np.arange(1,10)):
        #     #     print('[form factor] patch {}/{} ...'.format(i, self.__patch_count))
        #     # ffs = pool.starmap(self.__calculate_one_patch_form_factor, enumerate((self.__get_faces(self.mesh.faces))))
        #     ffs = pool.starmap(self.__calculate_one_patch_form_factor, enumerate(self.mesh.faces))
        #     # self.__calculate_one_patch_form_factor(0, self.mesh.faces[0,:])
        #     # ffs = pool.starmap(self.__calculate_one_patch_form_factor, enumerate(np.arange(0,100)))
        #     # pool.starmap(self.__calculate_one_patch_form_factor, enumerate(np.arange(0,100)))

        return ffs

    # def __call__(self, x):
    #     return self.__calculate_one_patch_form_factor(x)

    def get_number_of_rays(self):
        return self.__n_rays

    def __set_number_of_rays(self, number_of_rays):
        self.__n_rays = number_of_rays





    # def __init__(self, args, patch_list):
        # assert args.hemicube_edge > 0
        # print('Create hemicude with edge length {}'.format(args.hemicube_edge))
        #
        # self.edge = args.hemicube_edge
        # self.edge2 = args.hemicube_edge * 2
        # self.edge1_2 = args.hemicube_edge / 2
        # self.edge3_2 = args.hemicube_edge * 3 / 2
        # self.surface_area = self.edge**2 + 4 * self.edge * self.edge1_2
        #
        # self.patch_list = patch_list
        # self.patch_count = len(patch_list)
        #
        # self.delta_formfactor = np.zeros((self.edge2, self.edge2))
        # for i in range(self.edge2):
        #     for j in range(self.edge2):
        #         if not ((i < self.edge1_2 or self.edge3_2 < i) and (j < self.edge1_2 or self.edge3_2 < j)):
        #             x = (i + 0.5 - self.edge) / self.edge1_2
        #             y = (j + 0.5 - self.edge) / self.edge1_2
        #             z = 1.0
        #
        #             if x < -1:
        #                 z = 2 + x
        #                 x = -1.0
        #             if x > 1:
        #                 z = 2 - x
        #                 x = 1.0
        #             if y < -1:
        #                 z = 2 + y
        #                 y = -1.0
        #             if y > 1:
        #                 z = 2 - y
        #                 y = 1.0
        #
        #             self.delta_formfactor[i][j] = z / (np.pi * (x**2 + y**2 + z**2)**2) / self.surface_area
        #
        # # normalization
        # self.delta_formfactor /= np.sum(self.delta_formfactor)
    #
    # def calculate_form_factor(self, processes):
    #     with Pool(processes=processes) as pool:
    #         ffs = pool.starmap(self.calculate_one_patch_form_factor, enumerate(self.patch_list))
    #
    #     return np.array(ffs)
    #
    # def calculate_one_patch_form_factor(self, i, p_i):
    #     print('[form factor] patch {}/{} ...'.format(i, self.patch_count))
    #
    #     N_vs = []
    #     N_distance = []
    #     for j, p_j in enumerate(self.patch_list):
    #         if i == j:
    #             N_vs.append(np.array([[-1., -1.], [-1., -2.], [-2., -1.]]))
    #             N_distance.append(np.inf)
    #             continue
    #
    #         ci = p_i.center()
    #         cj = p_j.center()
    #
    #         v_ij = vectorize(ci, cj)
    #         n = p_i.normal
    #         if np.dot(v_ij, n) <= 0:
    #             N_vs.append(np.array([[-1., -1.], [-1., -2.], [-2., -1.]]))
    #             N_distance.append(np.inf)
    #             continue
    #
    #         transform = self.get_transform_matrix(p_i)
    #         if transform is None:
    #             N_vs.append(np.array([[-1., -1.], [-1., -2.], [-2., -1.]]))
    #             N_distance.append(np.inf)
    #             continue
    #         v0 = np.dot(vectorize(ci, p_j.vertices[0]), transform)
    #         v1 = np.dot(vectorize(ci, p_j.vertices[1]), transform)
    #         v2 = np.dot(vectorize(ci, p_j.vertices[2]), transform)
    #         v0 = self.project(v0)
    #         v1 = self.project(v1)
    #         v2 = self.project(v2)
    #         vs = np.array([v0, v1, v2])
    #
    #         N_vs.append(vs)
    #         N_distance.append(distance(ci, cj))
    #
    #     # collect all triangles and distances
    #     assert len(N_vs) == self.patch_count
    #     assert len(N_distance) == self.patch_count
    #     N_vs = np.array(N_vs)
    #     N_distance = np.array(N_distance)
    #
    #     # create empty form factor array
    #     ff = np.zeros(self.patch_count)
    #
    #     # speed up intersection test
    #     N_v0 = N_vs[:, 2] - N_vs[:, 0]
    #     N_v1 = N_vs[:, 1] - N_vs[:, 0]
    #     d00 = inner1d(N_v0, N_v0)
    #     d01 = inner1d(N_v0, N_v1)
    #     d11 = inner1d(N_v1, N_v1)
    #     invDenom = 1. / (d00 * d11 - d01 * d01)
    #
    #     # iterate hemicube
    #     for x in range(self.edge2):
    #         for y in range(self.edge2):
    #             # Barycentric Technique
    #             pnt_int = np.array([x + 0.5, y + 0.5])
    #             N_v2 = pnt_int - N_vs[:, 0]
    #             d02 = inner1d(N_v0, N_v2)
    #             d12 = inner1d(N_v1, N_v2)
    #             u = (d11 * d02 - d01 * d12) * invDenom
    #             v = (d00 * d12 - d01 * d02) * invDenom
    #
    #             # inside triangle
    #             within = (u >= 0.) & (v >= 0.) & (u + v < 1.)
    #             if within.any():
    #                 tmp_distance = N_distance.copy()
    #                 tmp_distance[~within] = np.inf
    #                 ff[np.argmin(tmp_distance)] += self.delta_formfactor[x][y]
    #
    #     return ff
    #
    # def get_transform_matrix(self, p):
    #     c = p.center()
    #     x = vectorize(c, p.vertices[0])
    #     x = np.multiply(x, 1 / np.linalg.norm(x))
    #     z = p.normal
    #     y = np.cross(z, x)
    #
    #     A = np.array([x, y, z])
    #     B = np.identity(3)
    #
    #     try:
    #         X = np.linalg.solve(A, B)
    #     except:
    #         X = None
    #     return X
    #
    # def project(self, v):
    #     x = v[0]
    #     y = v[1]
    #     z = v[2]
    #     if z < 0:
    #         z = 0.0
    #
    #     # side: right = 0, up = 1, left = 2, down = 3
    #     side = -1
    #     if x >= 0 and y >= 0:
    #         if x > y:
    #             side = 0
    #         else:
    #             side = 1
    #     elif x < 0 and y >= 0:
    #         if -x > y:
    #             side = 2
    #         else:
    #             side = 1
    #     elif x < 0 and y < 0:
    #         if -x > -y:
    #             side = 2
    #         else:
    #             side = 3
    #     else:
    #         if x > -y:
    #             side = 0
    #         else:
    #             side = 3
    #
    #     xy = np.sqrt(x**2 + y**2)
    #     theta = np.arctan(z / xy)
    #     if side == 0:
    #         if theta >= np.arctan(1 / np.sqrt((y / x)**2 + 1)):
    #             return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
    #         else:
    #             return self.edge - self.edge1_2 * (y / np.abs(x)), self.edge2 - self.edge1_2 * (z / np.abs(x))
    #     elif side == 1:
    #         if theta >= np.arctan(1 / np.sqrt((x / y)**2 + 1)):
    #             return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
    #         else:
    #             return self.edge1_2 * (z / np.abs(y)), self.edge + self.edge1_2 * (x / np.abs(y))
    #     elif side == 2:
    #         if theta >= np.arctan(1 / np.sqrt((y / x)**2 + 1)):
    #             return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
    #         else:
    #             return self.edge - self.edge1_2 * (y / np.abs(x)), self.edge1_2 * (z / np.abs(x))
    #     else:
    #         if theta >= np.arctan(1 / np.sqrt((x / y)**2 + 1)):
    #             return self.edge - self.edge1_2 * (y / np.abs(z)), self.edge + self.edge1_2 * (x / np.abs(z))
    #         else:
    #             return self.edge2 - self.edge1_2 * (z / np.abs(y)), self.edge + self.edge1_2 * (x / np.abs(y))
