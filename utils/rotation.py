import numpy as np
# from typing import overload
from multipledispatch import dispatch # for applying function overloading
from multiprocessing.dummy import Pool as ThreadPool
# import mgen
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
     Notes
     -----
     Check on https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
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


def rotation_matrices_from_angles(angles, directions):
    r"""
    Calculate a collection of rotation matrices defined by
    an input collection of rotation angles and rotation axes.
    Parameters
    ----------
    angles : ndarray
        Numpy array of shape (npts, ) storing a collection of rotation angles
    directions : ndarray
        Numpy array of shape (npts, 3) storing a collection of rotation axes in 3d
    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices
    Examples
    --------
    Create a set of random rotation matrices.
    >>> from halotools.utils.mcrotations import random_unit_vectors_3d
    >>> npts = int(1e4)
    >>> angles = np.random.uniform(-np.pi/2., np.pi/2., npts)
    >>> directions = random_unit_vectors_3d(npts)
    >>> rotation_matrices = rotation_matrices_from_angles(angles, directions)
    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 3d vectors
    check: https://github.com/astropy/halotools/blob/master/halotools/utils/rotations3d.py and https://stackoverflow.com/questions/47623582/efficiently-calculate-list-of-3d-rotation-matrices-in-numpy-or-scipy
    """
    directions = __normalized_vectors(directions)
    angles = np.atleast_1d(angles)
    npts = directions.shape[0]

    sina = np.sin(angles)
    cosa = np.cos(angles)

    R1 = np.zeros((npts, 3, 3))
    R1[:, 0, 0] = cosa
    R1[:, 1, 1] = cosa
    R1[:, 2, 2] = cosa

    R2 = directions[..., None] * directions[:, None, :]
    R2 = R2*np.repeat(1.-cosa, 9).reshape((npts, 3, 3))

    directions *= sina.reshape((npts, 1))
    R3 = np.zeros((npts, 3, 3))
    R3[:, [1, 2, 0], [2, 0, 1]] -= directions
    R3[:, [2, 0, 1], [1, 2, 0]] += directions

    return R1 + R2 + R3

def __normalized_vectors(vectors):
    r"""
    Return a unit-vector for each n-dimensional vector in the input list of n-dimensional points.
    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional points
    Returns
    -------
    normed_x : ndarray
        Numpy array of shape (npts, ndim)
    Examples
    --------
    Let's create a set of semi-random 3D unit vectors.
    >>> npts = int(1e3)
    >>> ndim = 3
    >>> x = np.random.random((npts, ndim))
    >>> normed_x = normalized_vectors(x)
    """

    vectors = np.atleast_2d(vectors)
    npts = vectors.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        return vectors/__elementwise_norm(vectors).reshape((npts, -1))

def __elementwise_norm(x):
    r"""
    Calculate the normalization of each element in a list of n-dimensional points.
    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional points
    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the norm of each n-dimensional point in x.
    Examples
    --------
    >>> npts = int(1e3)
    >>> ndim = 3
    >>> x = np.random.random((npts, ndim))
    >>> norms = elementwise_norm(x)
    """

    x = np.atleast_2d(x)
    return np.sqrt(np.sum(x**2, axis=1))


def __elementwise_dot(x, y):
    r"""
    Calculate the dot product between
    each pair of elements in two input lists of n-dimensional points.
    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional vectors
    y : ndarray
        Numpy array of shape (npts, ndim) storing a collection of n-dimensional vectors
    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the dot product between each
        pair of corresponding vectors in x and y.
    Examples
    --------
    Let's create two sets of semi-random 3D vectors, x1 and x2.

    >>> npts = int(1e3)
    >>> ndim = 3
    >>> x1 = np.random.random((npts, ndim))
    >>> x2 = np.random.random((npts, ndim))
    We then can find the dot product between each pair of vectors in x1 and x2.
    >>> dots = elementwise_dot(x1, x2)
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.sum(x * y, axis=1)


def rotation_matrices_from_vectors(v0, v1):
    r"""
    Calculate a collection of rotation matrices defined by two sets of vectors,
    v1 into v2, such that the resulting matrices rotate v1 into v2 about
    the mutually perpendicular axis.
    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, 3) storing a collection of initial vector orientations.
        Note that the normalization of `v0` will be ignored.
    v1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of final vectors.
        Note that the normalization of `v1` will be ignored.
    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) rotating each v0 into the corresponding v1
    Examples
    --------
    Create a set of random rotation matrices.
    >>> from halotools.utils.mcrotations import random_unit_vectors_3d
    >>> npts = int(1e4)
    >>> v0 = random_unit_vectors_3d(npts)
    >>> v1 = random_unit_vectors_3d(npts)
    >>> rotation_matrices = rotation_matrices_from_vectors(v0, v1)
    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 3d vectors
    """

    # # View inputs as arrays with at least two dimensions.
    # v0 = np.atleast_2d(v0)
    # v1 = np.atleast_2d(v1)
    #
    # # check whether both inputs have the same dimensions, if not make them
    # # TODO: maybe find a better way to check this
    # if v0.shape[0] > v1.shape[0]:
    #     v1 = np.tile(v1, (v0.shape[0], 1))
    # elif v0.shape[0] < v1.shape[0]:
    #     v0 = np.tile(v0, (v1.shape[0], 1))

    # v0 = __normalized_vectors(v0)
    # v1 = __normalized_vectors(v1)
    # directions = __vectors_normal_to_planes(v0, v1)
    # angles = __angles_between_list_of_vectors(v0, v1)
    #
    # # where angles are 0.0, replace directions with v0
    # mask_a = (np.isnan(directions[:,0]) | np.isnan(directions[:,1]) | np.isnan(directions[:,2]))
    # mask_b = (angles==0.0)
    # mask = mask_a | mask_b
    # directions[mask] = v0[mask]

    # return rotation_matrices_from_angles(angles, directions)

    rot = vvrotvec(v0, v1)
    return rotation_matrices_from_angles(rot[:,3], rot[:,0:3])


def __angles_between_list_of_vectors(v0, v1, tol=1e-3, vn=None):
    r""" Calculate the angle between a collection of n-dimensional vectors
    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-D vectors
        Note that the normalization of `v0` will be ignored.
    v1 : ndarray
        Numpy array of shape (npts, ndim) storing a collection of ndim-D vectors
        Note that the normalization of `v1` will be ignored.
    tol : float, optional
        Acceptable numerical error for errors in angle.
        This variable is only used to round off numerical noise that otherwise
        causes exceptions to be raised by the inverse cosine function.
        Default is 0.001.
    n1 : ndarray
        normal vector
    Returns
    -------
    angles : ndarray
        Numpy array of shape (npts, ) storing the angles between each pair of
        corresponding points in v0 and v1.
        Returned values are in units of radians spanning [0, pi].
    Examples
    --------
    Let's create two sets of semi-random 3D unit vectors.
    >>> npts = int(1e4)
    >>> ndim = 3
    >>> v1 = np.random.random((npts, ndim))
    >>> v2 = np.random.random((npts, ndim))
    We then can find the angle between each pair of vectors in v1 and v2.
    >>> angles = angles_between_list_of_vectors(v1, v2)
    """

    dot = __elementwise_dot(__normalized_vectors(v0), __normalized_vectors(v1))

    if vn is None:
        #  Protect against tiny numerical excesses beyond the range [-1 ,1]
        mask1 = (dot > 1) & (dot < 1 + tol)
        dot = np.where(mask1, 1., dot)
        mask2 = (dot < -1) & (dot > -1 - tol)
        dot = np.where(mask2, -1., dot)
        a = np.arccos(dot)
    else:
        cross = np.cross(v0, v1)
        a = np.arctan2(__elementwise_dot(cross, vn), dot)

    return a


def __vectors_normal_to_planes(x, y):
    r"""
    Given a collection of 3d vectors x and y, return a collection of
    3d unit-vectors that are orthogonal to x and y.
    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors
        Note that the normalization of `x` will be ignored.
    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors
        Note that the normalization of `y` will be ignored.
    Returns
    -------
    z : ndarray
        Numpy array of shape (npts, 3). Each 3d vector in z will be orthogonal
        to the corresponding vector in x and y.
    Examples
    --------
    Define a set of random 3D vectors
    >>> npts = int(1e4)
    >>> x = np.random.random((npts, 3))
    >>> y = np.random.random((npts, 3))
    now calculate a thrid set of vectors to a corresponding pair in `x` and `y`.
    >>> normed_z = angles_between_list_of_vectors(x, y)
    """
    return __normalized_vectors(np.cross(x, y))