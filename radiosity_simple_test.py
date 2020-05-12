import numpy as np
import numexpr as ne
import numpy.matlib
import matplotlib.pylab as plt
from scipy.linalg import norm
import time
# from numba import jit

# @jit(nopython=True)
def main():
    # start_time = time.time()

    # Further testing for building a radiosity lighting method for virtual
    # spaces. Here we pixelize all the walls and draw them with a smooth
    # gradient of gray values.
    #
    # Theodore Tsesmelis May 2020

    pi = np.pi

    # Choose mosaic resolution
    n = 30

    # Construct centerpoints of the mosaic tiles.
    # The letter d denotes the length of the side of a pixel.
    d = 2/n
    tmp = -1-d/2 + (np.arange(1,n+1))*d

    # Initialize centerpoint coordinate matrices
    width = n**2
    s = (width,5)
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
    F = np.zeros((5*width, 5*width))

    # # From the roof (jjj) to the back wall (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the back wall
    #         piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
    #         # Centerpoint of the current pixel in the roof
    #         pjjj = np.array([Xmat[jjj,1], Ymat[jjj,1], Zmat[jjj,1]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         r = np.linalg.norm(difvec)
    #         # r = norm(difvec)
    #         # View angles
    #         tmp2   = difvec/r
    #         cosjjj = abs(tmp2[2])
    #         cosiii = abs(tmp2[1])
    #         # Calculate element of F
    #         F[iii, width+jjj] = cosiii*cosjjj/(np.pi*(sc_par*r)**2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[2,:])
    cosiii = abs(tmp2[1,:])
    F[0:width,width+0:width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the floor (jjj) to the back wall (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the back wall
    #         piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
    #         # Centerpoint of the current pixel in the floor
    #         pjjj = np.array([Xmat[jjj,2], Ymat[jjj,2], Zmat[jjj,2]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[2])
    #         cosiii = abs(tmp2[1])
    #         # Calculate element of F
    #         F[iii, 2 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)
            
    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[2,:])
    cosiii = abs(tmp2[1,:])
    F[0:width,2*width+0:2*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the right-hand-side wall (jjj) to the back wall (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the back wall
    #         piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
    #         # Centerpoint of the current pixel in the right-hand-side wall
    #         pjjj = np.array([Xmat[jjj,3], Ymat[jjj,3], Zmat[jjj,3]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[1])
    #         # Calculate element of F
    #         F[iii, 3 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)
            
    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[1,:])
    F[0:width,3*width+0:3*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the left-hand-side wall (jjj) to the back wall (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the back wall
    #         piii = np.array([Xmat[iii,0], Ymat[iii,0], Zmat[iii,0]])
    #         # Centerpoint of the current pixel in the left-hand-side wall
    #         pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[1])
    #         # Calculate element of F
    #         F[iii, 4 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 0], Ymat[:, 0], Zmat[:, 0]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[1,:])
    F[0:width,4*width+0:4*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the floor (jjj) to the roof (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the roof
    #         piii = np.array([Xmat[iii,1], Ymat[iii,1], Zmat[iii,1]])
    #         # Centerpoint of the current pixel in the floor
    #         pjjj = np.array([Xmat[jjj, 2], Ymat[jjj, 2], Zmat[jjj, 2]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[2])
    #         cosiii = abs(tmp2[2])
    #         # Calculate element of F
    #         F[width + iii, 2 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    pjjj = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[2,:])
    cosiii = abs(tmp2[2,:])
    F[width+0:width+width,2*width+0:2*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the right-hand wall (jjj) to the roof (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the roof
    #         piii = np.array([Xmat[iii,1], Ymat[iii,1], Zmat[iii,1]])
    #         # Centerpoint of the current pixel in the right-hand wall
    #         pjjj = np.array([Xmat[jjj,3], Ymat[jjj,3], Zmat[jjj,3]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[2])
    #         # Calculate element of F
    #         F[width + iii, 3 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    pjjj = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[2,:])
    F[width+0:width+width, 3*width+0:3*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the left-hand-side wall (jjj) to the roof (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the roof
    #         piii = np.array([Xmat[iii,1], Ymat[iii,1], Zmat[iii,1]])
    #         # Centerpoint of the current pixel in the left-hand-side wall
    #         pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[2])
    #         # Calculate element of F
    #         F[width + iii, 4 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 1], Ymat[:, 1], Zmat[:, 1]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[2,:])
    F[width+0:width+width, 4*width+0:4*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the right-hand-side wall (jjj) to the floor (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the floor
    #         piii = np.array([Xmat[iii,2], Ymat[iii,2], Zmat[iii,2]])
    #         # Centerpoint of the current pixel in the right-hand-side wall
    #         pjjj = np.array([Xmat[jjj,3], Ymat[jjj,3], Zmat[jjj,3]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[2])
    #         # Calculate element of F
    #         F[2 * width + iii, 3 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    pjjj = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[2,:])
    F[2*width+0:2*width+width, 3*width+0:3*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the left-hand-side wall (jjj) to the floor (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the floor
    #         piii = np.array([Xmat[iii,2], Ymat[iii,2], Zmat[iii,2]])
    #         # Centerpoint of the current pixel in the left-hand-side wall
    #         pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[2])
    #         # Calculate element of F
    #         F[2 * width + iii, 4 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 2], Ymat[:, 2], Zmat[:, 2]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[2,:])
    F[2*width+0:2*width+width, 4*width+0:4*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # # From the left-hand-side wall (jjj) to the right-hand-side wall (iii)
    # for iii in range(0, width):
    #     for jjj in range(0, width):
    #         # Centerpoint of the current pixel in the right-hand-side wall
    #         piii = np.array([Xmat[iii,3], Ymat[iii,3], Zmat[iii,3]])
    #         # Centerpoint of the current pixel in the left-hand-side wall
    #         pjjj = np.array([Xmat[jjj, 4], Ymat[jjj, 4], Zmat[jjj, 4]])
    #         # Distance between the points
    #         difvec = piii - pjjj
    #         # r = np.linalg.norm(difvec)
    #         r = norm(difvec)
    #         # View angles
    #         tmp2 = difvec / r
    #         cosjjj = abs(tmp2[0])
    #         cosiii = abs(tmp2[0])
    #         # Calculate element of F
    #         F[3 * width + iii, 4 * width + jjj] = cosiii * cosjjj / (np.pi * (sc_par * r) ** 2)

    # Optimized version of above loops by using vectorization
    piii = np.array([Xmat[:, 3], Ymat[:, 3], Zmat[:, 3]]).T
    pjjj = np.array([Xmat[:, 4], Ymat[:, 4], Zmat[:, 4]]).T
    piii = piii[:, np.newaxis]
    difvec = np.transpose(ne.evaluate('piii - pjjj').reshape(-1, piii.shape[2]))
    r = np.linalg.norm(difvec, axis=0)
    tmp2 = ne.evaluate('difvec/r')
    cosjjj = abs(tmp2[0,:])
    cosiii = abs(tmp2[0,:])
    F[3*width+0:3*width+width, 4*width+0:4*width+width] = ne.evaluate('cosiii*cosjjj/(pi*(sc_par*r)**2)').reshape(width,-1)

    # elapsed_time = time.time() - start_time
    #
    # print('time cost = ', elapsed_time)

    # plt.spy(F)
    # plt.show()

    # Add the contribution of the area of each pixel
    F = ne.evaluate('(d**2)*F');

    # Use symmetry to finish the construction of F
    F = F + np.transpose(F);

    # plt.spy(F)
    # plt.show()
    
    # Construct the right hand side Evec of the radiosity equation. Evec
    # describes the contribution of emitted light in the scene. For example,
    # each pixel belonging to a lamp in the virtual space causes a positive
    # element in Evec.
    Evec = np.zeros(5*width)
    Evec[width + 1:width + 1 + width] = (np.array(np.sqrt((Xmat[:, 1] - 0.3) ** 2 + (Ymat[:, 1]) ** 2)) < 0.3)
    # indvec = np.full(Evec.shape, False)
    # indvec[width+1:width+1+width] = (np.array(np.sqrt((Xmat[:,1]-0.3)**2 + (Ymat[:,1])**2)) < 0.3)
    # Evec[indvec] = 1
    # Evec(n^2+round(n^2/2)-2) = 1;
    # Evec(3*n^2+round(n^2/2)-2) = 1;

    # Solve for color vector.
    # The parameter rho adjusts the surface material(how much incoming light
    # is reflected away from a patch, 0 < rho <= 1)
    rho = 1
    colorvec = np.dot(np.linalg.inv((np.eye(5 * n ** 2) - rho * F)), Evec.reshape(-1, 1))

    # Normalize the values of the color vector between 0 and 1.
    colorvec[colorvec < 0] = 0
    colorvec = colorvec / np.amax(colorvec)

    # The cut_param below is for preventing the lamp color being too far away
    # from the gray colors of everything else. Setting cut_param = 1; has no
    # effect, and setting 0 < cut_param < 1 can improve the plot. Experiment
    # with it to find the best value.
    cut_param = .012
    colorvec[colorvec > cut_param] = cut_param
    colorvec = colorvec / np.amax(colorvec)

    # Gamma correction for optimal gray levels.Choosing gammacorr = 1; has no
    # effect, and taking 0 < gammacorr = < 1 small will lighten up the dark
    # shades. 
    gammacorr = .7
    colorvec = colorvec**(gammacorr)

    # Construct grayscale color matrix by repeating the same color vector for
    # red, green and blue channels.
    # colormat = repmat(colorvec(:), 1, 3);
    colormat = np.matlib.repmat(colorvec, 1, 3)

    print()

if __name__ == "__main__":
    main()
    print("End!!!")