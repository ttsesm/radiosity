import numpy as np
# from typing import overload
from multipledispatch import dispatch # for applying function overloading
from multiprocessing.dummy import Pool as ThreadPool
import mgen
import trimesh


def unit_vector(vector, maxzero=1e-12):
    """ Returns the unit vector of the vector. [Vectorized version] """
    unit_vec = np.zeros(vector.shape)
    # return vector / np.linalg.norm(vector) # initial implementation, not checking for 0 norm, fixed below
    norm_vec = np.linalg.norm(vector, axis=1)
    i = norm_vec > maxzero
    if i.any():
        unit_vec[i] = vector[i] / norm_vec[i,np.newaxis]

    return unit_vec
    # if norm_vec <= maxzero:
    #     return np.zeros(np.array(vector).shape)
    # else:
    #     return vector / norm_vec

def sl3dnormalize(vec, maxzero=1e-12):
    """ Returns the unit vector of the vector, same as unit_vector(). Ported from Matlab  """
    norm_vec = np.linalg.norm(vec, axis=1)
    if norm_vec <= maxzero:
        return np.zeros(np.array(vec).shape)
    else:
        return vec / norm_vec

def vvrotvec(v1, v2):
    """ Calculates a rotation needed to transform a 3d vector A to a 3d vector B.
    The result R is a 4-element axis-angle rotation row vector. First three elements
    specify the rotation axis, the last element defines the angle of rotation in radians
    between vectors 'v1' and 'v2'::

            >>> vvrotvec((1, 0, 0), (0, 1, 0))
            0.0 0.0 1.0 1.5707963267948966
            >>> vvrotvec((1, 0, 0), (1, 0, 0))
            0.0 0.0 1.0 0.0
            >>> vvrotvec((1, 0, 0), (-1, 0, 0))
            0.0 0.0 1.0 3.141592653589793
    """
    # View inputs as arrays with at least two dimensions.
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)

    # check whether both inputs have the same dimensions, if not make them
    # TODO: maybe find a better way to check this
    if v1.shape[0] > v2.shape[0]:
        v2 = np.tile(v2, (v1.shape[0], 1))
    elif v1.shape[0] < v2.shape[0]:
        v1 = np.tile(v1, (v2.shape[0], 1))

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    # test = sl3dnormalize(np.cross(v1_u, v2_u))

    ax = unit_vector(np.cross(v1_u, v2_u))
    dot_product = np.einsum('ij,ij->i', v1_u, v2_u)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # if cross(v1_u, v2_u) is zero, vectors are parallel (angle = 0) or antiparallel
    # (angle = pi). In both cases it is necessary to provide a valid axis. Let's
    # select one that satisfies both cases - an axis that is perpendicular to
    # both vectors. We find this vector by cross product of the first vector
    # with the "least aligned" basis vector.
    empty_rows = np.all(np.isclose(ax, 0), axis=1)
    if empty_rows.any():
        indices_of_empty_rows = np.where(empty_rows)
        absv1_u = abs(v1_u[indices_of_empty_rows])
        mind = np.argmin(absv1_u,axis=1)
        # c = np.zeros(3)
        c = np.zeros(shape=(mind.shape[0],3))
        # c[mind] = 1
        c[:,mind[0]] = 1
        ax[[indices_of_empty_rows]] = np.cross(v1_u[indices_of_empty_rows], c)

    # return np.hstack([ax, angle])
    return np.column_stack([ax, angle])

# @overload
@dispatch(object, object)
# def vrrotvec2mat(v1,v2, processes=4):
def vrrotvec2mat(v1,v2, **kwargs):
    processes = kwargs.pop('processes', 4)
    rot = vvrotvec(v1,v2)

    # matrix = rotation_matrices([0], [0,1,0])
    # # matrix = rotation_matrices((0), (0,1,0))

    return vrrotvec2mat(rot, processes=processes)

@dispatch(object)
# def vrrotvec2mat(r, processes=4):
def vrrotvec2mat(r, **kwargs):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, r = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |

     The rotation vector r is a row vector of 4 elements, where the first three elements specify the
     rotation axis and the last element defines the angle.
    """
    processes = kwargs.pop('processes', 4)
    with ThreadPool(processes=processes) as pool:
        mms = pool.starmap(__compute_rotation_matrix, enumerate(r))
    # __compute_rotation_matrix(0,r)

        return np.array(mms)

def __compute_rotation_matrix(i,r):
    r = r.flatten()
    # build the rotation matrix
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c
    n = unit_vector(r[0:3].reshape(-1,3)).flatten()
    x = n[0]
    y = n[1]
    z = n[2]
    m = np.zeros((3, 3))
    # Calculate the rotation matrix elements.
    m[0, 0] = 1.0 + (1.0 - c) * (x ** 2 - 1.0)
    m[0, 1] = -z * s + (1.0 - c) * x * y
    m[0, 2] = y * s + (1.0 - c) * x * z
    m[1, 0] = z * s + (1.0 - c) * x * y
    m[1, 1] = 1.0 + (1.0 - c) * (y ** 2 - 1.0)
    m[1, 2] = -x * s + (1.0 - c) * y * z
    m[2, 0] = -y * s + (1.0 - c) * x * z
    m[2, 1] = x * s + (1.0 - c) * y * z
    m[2, 2] = 1.0 + (1.0 - c) * (z ** 2 - 1.0)
    return m