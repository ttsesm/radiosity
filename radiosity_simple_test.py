import numpy as np
import matplotlib.pylab as plt
# from scipy import *
from scipy.linalg import norm
import time
# from numba import jit

# @jit(nopython=True)
def main():
    start_time = time.time()
    # Further testing for building a radiosity lighting method for virtual
    # spaces. Here we pixelize all the walls and draw them with a smooth
    # gradient of gray values.
    #
    # Theodore Tsesmelis May 2020

    # Choose mosaic resolution
    n = 30

    # Construct centerpoints of the mosaic tiles.
    # The letter d denotes the length of the side of a pixel.
    d = 2/n
    tmp = -1-d/2 + (np.arange(1,n+1))*d

    # Initialize centerpoint coordinate matrices
    s = (n**2,5)
    Xmat = np.zeros(s)
    Ymat = np.zeros(s)
    Zmat = np.zeros(s)

    # Construct the centerpoints for all the tiles in all the five walls.
    # The ordering of the five walls below fixes the indexing of all the tiles
    # using just one number running from 1 to 5*(n^2).

    # The back wall
    X, Z= np.meshgrid(tmp, tmp, sparse=False, indexing='ij')
    Xmat[:,0] = X.flatten()
    Zmat[:,0] = Z.flatten()
    Ymat[:,0] = 1

    # Roof
    X, Y = np.meshgrid(tmp, tmp, sparse=False, indexing='ij')
    Xmat[:,1] = X.flatten()
    Ymat[:,1] = Y.flatten()
    Zmat[:,1] = 1

    # Floor
    Xmat[:,2] = X.flatten()
    Ymat[:,2] = Y.flatten()
    Zmat[:,2] = -1

    # Right-hand-side wall
    Y, Z = np.meshgrid(tmp, tmp, sparse=False, indexing='ij')
    Ymat[:,3] = Y.flatten()
    Zmat[:,3] = Z.flatten()
    Xmat[:,3] = 1

    # Left-hand-side wall
    Ymat[:,4] = Y.flatten()
    Zmat[:,4] = Z.flatten()
    Xmat[:,4] = -1

    # Construct the color vector (B-vector) using the radiosity lighting model.

    # Scaling parameter for the inverse square attenuation law.
    # Taking sc_par = 1; leads to unnatural lighting, taking sc_par > 2 has
    # better scaling between the size of the cubic room and the attenuation
    # law.
    sc_par = 3;

    # Form the geometrical view factor matrix F. Note that F is symmetric.
    # See http://en.wikipedia.org/wiki/View_factor for details of computation.

    # Initialize the matrix
    F = np.zeros((5*n**2, 5*n**2))

    # From the roof (jjj) to the back wall (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the back wall
            piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
            # Centerpoint of the current pixel in the roof
            pjjj = np.array([Xmat[jjj,1], Ymat[jjj,1], Zmat[jjj,1]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2   = difvec/r
            cosjjj = abs(tmp2[2])
            cosiii = abs(tmp2[1])
            # Calculate element of F
            F[iii, n**2+jjj] = cosiii*cosjjj/(np.pi*(sc_par*r)**2)

    # From the floor (jjj) to the back wall (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the back wall
            piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
            # Centerpoint of the current pixel in the floor
            pjjj = np.array([Xmat[jjj,2], Ymat[jjj,2], Zmat[jjj,2]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[2])
            cosiii = abs(tmp2[1])
            # Calculate element of F
            F[iii, 2 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the right-hand-side wall (jjj) to the back wall (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the back wall
            piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
            # Centerpoint of the current pixel in the right-hand-side wall
            pjjj = np.array([Xmat[jjj,3], Ymat[jjj,3], Zmat[jjj,3]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[1])
            # Calculate element of F
            F[iii, 3 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the left-hand-side wall (jjj) to the back wall (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the back wall
            piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
            # Centerpoint of the current pixel in the left-hand-side wall
            pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[1])
            # Calculate element of F
            F[iii, 4 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the floor (jjj) to the roof (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the roof
            piii = np.array([Xmat[iii,1], Ymat[iii,1], Zmat[iii,1]])
            # Centerpoint of the current pixel in the floor
            pjjj = np.array([Xmat[jjj, 2], Ymat[jjj, 2], Zmat[jjj, 2]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[2])
            cosiii = abs(tmp2[2])
            # Calculate element of F
            F[n ** 2 + iii, 2 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the right-hand wall (jjj) to the roof (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the roof
            piii = np.array([Xmat[iii,1], Ymat[iii,1], Zmat[iii,1]])
            # Centerpoint of the current pixel in the right-hand wall
            pjjj = np.array([Xmat[jjj,3], Ymat[jjj,3], Zmat[jjj,3]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[2])
            # Calculate element of F
            F[n ** 2 + iii, 3 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the left-hand-side wall (jjj) to the roof (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the roof
            piii = np.array([Xmat[iii,1], Ymat[iii,1], Zmat[iii,1]])
            # Centerpoint of the current pixel in the left-hand-side wall
            pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[2])
            # Calculate element of F
            F[n ** 2 + iii, 4 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the right-hand-side wall (jjj) to the floor (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the floor
            piii = np.array([Xmat[iii,2], Ymat[iii,2], Zmat[iii,2]])
            # Centerpoint of the current pixel in the right-hand-side wall
            pjjj = np.array([Xmat[jjj,3], Ymat[jjj,3], Zmat[jjj,3]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[2])
            # Calculate element of F
            F[2 * n ** 2 + iii, 3 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the left-hand-side wall (jjj) to the floor (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the floor
            piii = np.array([Xmat[iii,2], Ymat[iii,2], Zmat[iii,2]])
            # Centerpoint of the current pixel in the left-hand-side wall
            pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[2])
            # Calculate element of F
            F[2 * n ** 2 + iii, 4 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # From the left-hand-side wall (jjj) to the right-hand-side wall (iii)
    for iii in range(0, n**2):
        for jjj in range(0, n**2):
            # Centerpoint of the current pixel in the right-hand-side wall
            piii = np.array([Xmat[iii,3], Ymat[iii,3], Zmat[iii,3]])
            # Centerpoint of the current pixel in the left-hand-side wall
            pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
            # Distance between the points
            difvec = piii - pjjj
            # r = np.linalg.norm(difvec)
            r = norm(difvec)
            # View angles
            tmp2 = difvec / r
            cosjjj = abs(tmp2[0])
            cosiii = abs(tmp2[0])
            # Calculate element of F
            F[3 * n ** 2 + iii, 4 * n ** 2 + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # betterspy.show(F)

    elapsed_time = time.time() - start_time

    print('time cost = ', elapsed_time)

    plt.spy(F)
    plt.show()

if __name__ == "__main__":
    main()
    print("End!!!")