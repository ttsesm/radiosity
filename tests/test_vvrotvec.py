# import time
# import multiprocessing
# import os
# import sys
# import pyvista as pv

import numpy as np
from utils import rotation as r

def test_vvrotvec():
############### test case 1x3 vector vs 1x3 vector
    rot = r.vvrotvec([0, 0, 0], [0, 1, 0])

    M = r.vrrotvec2mat(rot)

    print('Test case 1x3 vs 1x3:')
    print('Rotation vector: {}\n'.format(rot))
    print('Rotation matrix: {}\n'.format(M))



################ test case nx3 vector vs nx3 vector
    rot = r.vvrotvec(np.vstack([[0, 0, 0],[0, 0, 0]]), np.vstack([[0, 1, 0],[0, 1, 0]]))

    M = r.vrrotvec2mat(rot)

    print('Test case nx3 vs nx3:')
    print('Rotation vector: {}\n'.format(rot))
    print('Rotation matrix: {}\n'.format(M))



################ test case 1x3 vector vs nx3 vector
    rot = r.vvrotvec([0, 0, 0], np.vstack([[0, 1, 0],[0, 1, 0]]))

    M = r.vrrotvec2mat(rot)

    print('Test case 1x3 vs nx3:')
    print('Rotation vector: {}\n'.format(rot))
    print('Rotation matrix: {}\n'.format(M))



################ test case nx3 vector vs 1x3 vector
    rot = r.vvrotvec(np.vstack([[0, 0, 0],[0, 0, 0]]), [0, 1, 0])

    M = r.vrrotvec2mat(rot)

    print('Test case nx3 vs 1x3:')
    print('Rotation vector: {}\n'.format(rot))
    print('Rotation matrix: {}\n'.format(M))



def test_vrrotvec2mat():

################ test case overload rotation matrix from 1x3 vector vs 1x3 vector
    M = r.vrrotvec2mat([0, 0, 0], [0, 1, 0])
    print('Rotation matrix: {}\n'.format(M))


################ test case overload rotation matrix from nx3 vector vs nx3 vector
    M = r.vrrotvec2mat(np.vstack([[0, 0, 0],[0, 0, 0]]), np.vstack([[0, 1, 0],[0, 1, 0]]))
    print('Rotation matrix: {}\n'.format(M))

################ test case overload rotation matrix from 1x3 vector vs nx3 vector
    M = r.vrrotvec2mat([0, 0, 0], np.vstack([[0, 1, 0],[0, 1, 0]]))
    print('Rotation matrix: {}\n'.format(M))

################ test case nx3 vector vs 1x3 vector
    M = r.vrrotvec2mat(np.vstack([[0, 0, 0],[0, 0, 0]]), [0, 1, 0])
    print('Rotation matrix: {}\n'.format(M))

if __name__ == '__main__':
    print('Testing the rotation module!!!!')

    test_vvrotvec()
    test_vrrotvec2mat()
    print("exiting main")