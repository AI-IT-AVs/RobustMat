################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def build_se3_transform_batch(xyzrpy_batch): ### size Nx6 
    N = xyzrpy_batch.shape[0]
    if xyzrpy_batch.shape[1] != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = np.zeros((N,4,4))
    idx = np.arange(4)
    se3[:, idx, idx] = 1  
    se3[:, 0:3, 0:3] = euler_to_so3_batch(xyzrpy_batch[:,3:6])
    se3[:, 0:3, 3] =  xyzrpy_batch[:,0:3]
    
    return se3

def build_xyzrpy_transform_batch(se3_batch): ### size Nx4x4 
    N = se3_batch.shape[0]
    if se3_batch.shape[1] != 4 or se3_batch.shape[1]!=4:
        raise ValueError("Must supply 4x4 values to build transform")

    xyzrpy_batch = np.zeros((N,6))
    for i, se3 in enumerate(se3_batch):
        xyzrpy_batch[i,3:]  = so3_to_euler(se3_batch[i,0:3,0:3])
        xyzrpy_batch[i,0:3] = se3_batch[i,0:3,3]
    
    return xyzrpy_batch



def euler_to_so3_batch(rpy_batch):  ### size Nx3 
    N = rpy_batch.shape[0]
    if rpy_batch.shape[1] != 3:
        raise ValueError("Euler angles must have three components")

    
    sinrpy = np.sin(rpy_batch)
    cosrpy = np.cos(rpy_batch)
    
    R_x_batch = np.zeros((N, 3,3))
    R_x_batch[:,0,0] = 1
    R_x_batch[:,1,1] = cosrpy[:,0]
    R_x_batch[:,1,2] = -sinrpy[:,0]
    R_x_batch[:,2,1] = sinrpy[:,0]
    R_x_batch[:,2,2] = cosrpy[:,0]
    
    R_y_batch = np.zeros((N, 3,3))
    R_y_batch[:,0,0] = cosrpy[:,1]
    R_y_batch[:,0,2] = sinrpy[:,1]
    R_y_batch[:,1,1] = 1
    R_y_batch[:,2,0] = -sinrpy[:,1]
    R_y_batch[:,2,2] = cosrpy[:,1]   
    
    R_z_batch = np.zeros((N, 3,3))
    R_z_batch[:,0,0] = cosrpy[:,2]
    R_z_batch[:,0,1] = -sinrpy[:,2]
    R_z_batch[:,1,0] = sinrpy[:,2]
    R_z_batch[:,1,1] = cosrpy[:,2]   
    R_z_batch[:,2,2] = 1
    
    R_zy = np.einsum('...jk,...kh->...jh', R_z_batch, R_y_batch)
    R_zyx = np.einsum('...jk,...kh->...jh', R_zy, R_x_batch)
    
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy

def get_transform_to_utm(direction, location):
    transform_to_utm = np.eye(4)
    transform_to_utm[0:3, 0:3] = direction
    transform_to_utm[0:3, 3] = np.squeeze(np.asarray(location))
    
    return transform_to_utm


def get_transform_from_utm(direction, location): ##from utm to vehicle_view 
#     print(location)
#     print("nnnnnnnnn")
    transform_to_utm = get_transform_to_utm(direction, location)
    trans = np.eye(4)
    rot = np.transpose(transform_to_utm[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_utm[0:3, 3])
    return trans



def get_transform_to_utm_batch(direction, location):
    N = direction.shape[0]
    transform_to_utm = np.zeros((N,4,4))
    idx = np.arange(4)
    transform_to_utm[:, idx, idx] = 1 
    
    
    transform_to_utm[:, 0:3, 0:3] = direction
    transform_to_utm[:, 0:3, 3] = location
    
    return transform_to_utm

def get_transform_from_utm_batch(direction, location): ##from utm to vehicle_view 
    transform_to_utm = get_transform_to_utm_batch(direction, location)
    trans = np.zeros((transform_to_utm.shape[0],4,4))
    idx = np.arange(4)
    trans[:, idx, idx] = 1 
    
    rot = np.transpose(transform_to_utm[:, 0:3, 0:3],(0,2,1))
    trans[:, 0:3, 0:3] = rot
    trans[:, 0:3, 3] = np.einsum('...jk,...k->...j', rot, -transform_to_utm[:, 0:3, 3])
    return trans