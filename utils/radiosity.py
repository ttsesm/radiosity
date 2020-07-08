#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
# from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import numpy_indexed as npi
# from numpy.core.umath_tests import inner1d

import scipy

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


class Radiosity(object):
    def __init__(self, ffs, rho, e):

        # self.F = ffs # form factors matrix
        self.rho = rho # reflectance information
        self.e = e # self emmitance, describes the contribution of emitted light in the scene from the patches belonging to light sources

        nbf = ffs.shape[0]

        if isinstance(rho, int):
            ffs = rho * ffs
        else:
            ffs = np.diag(rho) * ffs

        self.F = np.eye(nbf) - ffs

        self.r = self.h = self.q = np.array([])

        return



    def solve(self, method=0, itermax=30):

        if method == 0:
            self.r = self.__direct_radiosity_solution()
        elif method == 1:
            self.r = self.__iterative_radiosity_solution(itermax)

        # flux radiatif incident
        # self.h = (self.r - self.e) / self.rho; # original equation
        self.h = ((self.r - self.e) * np.pi) / self.rho # add Lambertian assumption

        #flux sortant
        self.q = self.r - self.h

        return self.r, self.h, self.q


    def __direct_radiosity_solution(self):

        return self.__solve_minnonzero(self.F, self.e)

    def __iterative_radiosity_solution(self, itermax):

        R = copy.deepcopy(self.e)
        norme = 0.001 * np.linalg.norm(self.e)
        for i in range(itermax):
            res = self.e - self.F @ R
            nres = np.linalg.norm(res)
            print('Iteration {} : residual: {} \n'.format(i, nres))
            R += res
            if nres < norme:
                break

        return R

    def __solve_minnonzero(self, A, b):
        '''
        Check on:
        1. https://pythonquestion.com/post/how-can-i-obtain-the-same-special-solutions-to-underdetermined-linear-systems-that-matlab-s-a-b-mldivide-operator-returns-using-numpy-scipy/
        2. https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
        3. https://stackoverflow.com/questions/38286740/matlab-backslash-mldivide-versus-numpy-lstsq-with-rectangular-matrix?noredirect=1&lq=1
        4. https://stackoverflow.com/questions/25001753/numpys-linalg-solve-and-linalg-lstsq-not-giving-same-answer-as-matlabs
        '''
        x1, res, rnk, s = np.linalg.lstsq(A, b)
        if rnk == A.shape[1]:
            return x1  # nothing more to do if A is full-rank
        Q, R, P = scipy.linalg.qr(A.T, mode='full', pivoting=True)
        Z = Q[:, rnk:].conj()
        C = np.linalg.solve(Z[rnk:], -x1[rnk:])
        return x1 + Z.dot(C)