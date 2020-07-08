import numpy as np
import numexpr as ne
import numpy_indexed as npi
import os
import copy
import multiprocessing
from utils import Isocell
from utils import LightDistributionCurve
from utils import FormFactor
from utils import Radiosity

from scipy.io import loadmat

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import pyvista as pv
import open3d as o3d
import trimesh
import pyembree
import vtkplotter as vp

from numba import jit, cuda


def test_radiosity():
    pi = np.pi

    # Choose mosaic resolution
    n = 30

    # Construct centerpoints of the mosaic tiles.
    # The letter d denotes the length of the side of a pixel.
    d = 2 / n
    tmp = -1 - d / 2 + (np.arange(1, n + 1)) * d

    # Initialize centerpoint coordinate matrices
    width = n ** 2
    s = (width, 5)
    Xmat = np.zeros(s)
    Ymat = np.zeros(s)
    Zmat = np.zeros(s)

    # Construct the centerpoints for all the tiles in all the five walls.
    # The ordering of the five walls below fixes the indexing of all the tiles
    # using just one number running from 1 to 5*(n^2).

    # The back wall
    X, Z = np.meshgrid(tmp, tmp, sparse=False, indexing='ij')
    Xmat[:, 0] = X.flatten()
    Zmat[:, 0] = Z.flatten()
    Ymat[:, 0] = 1

    # Roof
    X, Y = np.meshgrid(tmp, tmp, sparse=False, indexing='ij')
    Xmat[:, 1] = X.flatten()
    Ymat[:, 1] = Y.flatten()
    Zmat[:, 1] = 1

    # Floor
    Xmat[:, 2] = X.flatten()
    Ymat[:, 2] = Y.flatten()
    Zmat[:, 2] = -1

    # Right-hand-side wall
    Y, Z = np.meshgrid(tmp, tmp, sparse=False, indexing='ij')
    Ymat[:, 3] = Y.flatten()
    Zmat[:, 3] = Z.flatten()
    Xmat[:, 3] = 1

    # Left-hand-side wall
    Ymat[:, 4] = Y.flatten()
    Zmat[:, 4] = Z.flatten()
    Xmat[:, 4] = -1

    # Construct the color vector (B-vector) using the radiosity lighting model.

    # Scaling parameter for the inverse square attenuation law.
    # Taking sc_par = 1; leads to unnatural lighting, taking sc_par > 2 has
    # better scaling between the size of the cubic room and the attenuation
    # law.
    sc_par = 3;

    # Form the geometrical view factor matrix F. Note that F is symmetric.
    # See http://en.wikipedia.org/wiki/View_factor for details of computation.

    # Initialize the matrix
    F = np.zeros((5 * width, 5 * width))

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[2, :])
    cosiii = abs(tmp2[1, :])
    F[0:width, width + 0:width + width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[2, :])
    cosiii = abs(tmp2[1, :])
    F[0:width, 2 * width + 0:2 * width + width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[1, :])
    F[0:width, 3 * width + 0:3 * width + width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[1, :])
    F[0:width, 4 * width + 0:4 * width + width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    pjjj = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[2, :])
    cosiii = abs(tmp2[2, :])
    F[width + 0:width + width, 2 * width + 0:2 * width + width] = ne.evaluate(
        'cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    pjjj = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[2, :])
    F[width + 0:width + width, 3 * width + 0:3 * width + width] = ne.evaluate(
        'cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[2, :])
    F[width + 0:width + width, 4 * width + 0:4 * width + width] = ne.evaluate(
        'cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    pjjj = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[2, :])
    F[2 * width + 0:2 * width + width, 3 * width + 0:3 * width + width] = ne.evaluate(
        'cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[2, :])
    F[2 * width + 0:2 * width + width, 4 * width + 0:4 * width + width] = ne.evaluate(
        'cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0, :])
    cosiii = abs(tmp2[0, :])
    F[3 * width + 0:3 * width + width, 4 * width + 0:4 * width + width] = ne.evaluate(
        'cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width, -1)

    # Add the contribution of the area of each pixel
    F = ne.evaluate('(d**2)*F');

    # Use symmetry to finish the construction of F
    F = F + np.transpose(F);

    # Construct the right hand side Evec of the radiosity equation. Evec
    # describes the contribution of emitted light in the scene. For example,
    # each pixel belonging to a lamp in the virtual space causes a positive
    # element in Evec.
    Evec = np.zeros(5 * width)
    Evec[width :width + width] = (np.array(np.sqrt((Xmat[:, 1] - 0.3) ** 2 + (Ymat[:, 1]) ** 2)) < 0.3)
    # indvec = np.full(Evec.shape, False)
    # indvec[width+1:width+1+width] = (np.array(np.sqrt((Xmat[:,1]-0.3)**2 + (Ymat[:,1])**2)) < 0.3)
    # Evec[indvec] = 1
    # Evec(n^2+round(n^2/2)-2) = 1;
    # Evec(3*n^2+round(n^2/2)-2) = 1;

    # Solve for color vector.
    # The parameter rho adjusts the surface material(how much incoming light
    # is reflected away from a patch, 0 < rho <= 1)
    rho = 1

    del X, Xmat, Y, Ymat, Z, Zmat, cosiii, cosjjj, piii, pjjj, tmp, tmp2, d, difvec, n, pi, r, sc_par, s, width

    r = Radiosity(F, rho, Evec).solve(method=0)[0]

    print('End testing the radiosity module!!!!')



if __name__ == '__main__':
    print('Testing the radiosity module!!!!')
    test_radiosity()

    os._exit(0)