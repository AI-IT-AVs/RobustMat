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

import bisect
import csv
import numpy as np
import numpy.matlib as ml
from transform import *
import time
import pandas as pd


def interpolate_vo_poses(vo_path, pose_timestamps, origin_timestamp):
    """Interpolate poses from visual odometry.

    Args:
        vo_path (str): path to file containing relative poses from visual odometry.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(vo_path) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)

        vo_timestamps = [0]
        abs_poses = [ml.identity(4)]

        lower_timestamp = min(min(pose_timestamps), origin_timestamp)
        upper_timestamp = max(max(pose_timestamps), origin_timestamp)

        for row in vo_reader:
            timestamp = int(row[0])
            if timestamp < lower_timestamp:
                vo_timestamps[0] = timestamp
                continue

            vo_timestamps.append(timestamp)

            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            abs_poses.append(abs_pose)

            if timestamp >= upper_timestamp:
                break

    return interpolate_poses(vo_timestamps, abs_poses, pose_timestamps, origin_timestamp)


def interpolate_ins_poses1(ins_path, pose_timestamps, origin_timestamp, use_rtk=False):
    """Interpolate poses from INS.

    Args:
        ins_path (str): a pandas file (not path)
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    start = time.time()
    ###input a file instead of a path (to save time when read it frequently)
    if isinstance(ins_path, str):
         ins_reader = pd.read_csv(ins_path)
    else:
         ins_reader = ins_path
            
    all_times = ins_reader['timestamp'].to_numpy()


    upper_timestamp = max(max(pose_timestamps), origin_timestamp)+1e5
    lower_timestamp = min(min(pose_timestamps), origin_timestamp)-1e6
    idx = np.logical_and(all_times>lower_timestamp, all_times<upper_timestamp)
    ins_timestamps = all_times[idx]
    abs_poses_euler =  ins_reader.iloc[idx,[5,6,7,-3,-2,-1]].to_numpy()
    
    abs_poses = build_se3_transform_batch(abs_poses_euler)
    end = time.time()
    poses, poses_origin_time = interpolate_poses(ins_timestamps, abs_poses, pose_timestamps, origin_timestamp)
    
    return poses, poses_origin_time, ins_timestamps, abs_poses_euler


npzfile = np.load('radar_gt_2019-01-18-14-14-42.npz')

radar_loc = npzfile['radar_loc']
radar_timesteps = npzfile['radar_timesteps']
radar_loc_xyzrpy = npzfile['radar_loc_xyzrpy']


def interpolate_ins_poses(ins_path, pose_timestamps, origin_timestamp, use_rtk=False):
    
    all_times = radar_timesteps
    upper_timestamp = max(max(pose_timestamps), origin_timestamp)+1e6
    lower_timestamp = min(min(pose_timestamps), origin_timestamp)-1e6
    idx = np.logical_and(all_times>lower_timestamp, all_times<upper_timestamp)
    ins_timestamps = all_times[idx]
    abs_poses = radar_loc[idx]
    
    poses, poses_origin_time = interpolate_poses(ins_timestamps, abs_poses, pose_timestamps, origin_timestamp)
    
    return poses, poses_origin_time, ins_timestamps, radar_loc_xyzrpy[idx]

def interpolate_poses(ins_timestamps, abs_poses, requested_timestamps, origin_timestamp):
        requested_timestamps = requested_timestamps.copy()
        requested_timestamps.insert(0, origin_timestamp)
        requested_timestamps = np.asarray(requested_timestamps)
        abs_poses = build_xyzrpy_transform_batch(abs_poses)
        
        
        t1_ind = np.sum(ins_timestamps.reshape(1,-1) <= requested_timestamps.reshape(-1,1), axis=1)
        t2_ind = t1_ind + 1
        
        point_1_x = abs_poses[t1_ind,0]
        point_1_y = abs_poses[t1_ind,1]
        point_1_z = abs_poses[t1_ind,2]
        point_1_roll = abs_poses[t1_ind,3]
        point_1_pitch = abs_poses[t1_ind,4]
        point_1_yaw = abs_poses[t1_ind,5]
        
        point_2_x = abs_poses[t2_ind, 0]
        point_2_y = abs_poses[t2_ind, 1]
        point_2_z = abs_poses[t2_ind, 2]
        point_2_roll = abs_poses[t2_ind, 3]
        point_2_pitch = abs_poses[t2_ind, 4]
        point_2_yaw = abs_poses[t2_ind, 5]
        
        T_21 = ins_timestamps[t2_ind] - ins_timestamps[t1_ind]
        t_x1 = requested_timestamps - ins_timestamps[t1_ind]

        x = point_1_x + (t_x1/T_21)*(point_2_x - point_1_x)
        y = point_1_y + (t_x1/T_21)*(point_2_y - point_1_y)
        z = point_1_z + (t_x1/T_21)*(point_2_z - point_1_z)
        roll = point_1_roll + (t_x1/T_21)*(point_2_roll - point_1_roll)
        pitch = point_1_pitch + (t_x1/T_21)*(point_2_pitch - point_1_pitch)
        yaw = point_1_yaw + (t_x1/T_21)*(point_2_yaw - point_1_yaw)
        
        
        gps_vehicle = build_se3_transform_batch(np.stack([x,y,z,roll,pitch,yaw], axis=1))
        gps_origin_time2utm = gps_vehicle[0,...]
        
        gps2_origin_time = get_transform_from_utm(gps_vehicle[0, 0:3,0:3], gps_vehicle[0, 0:3, 3].reshape(-1))
        gps_requested_timestamps = gps_vehicle[1:,...]
        
        poses_out = np.einsum('...jk,...kh->...jh', gps2_origin_time, gps_requested_timestamps)##related to the selected frame at origin_time
        
        return  [i for i in poses_out], gps_origin_time2utm
    

def interpolate_poses_old(pose_timestamps, abs_poses, requested_timestamps, origin_timestamp):
    """Interpolate between absolute poses.

    Args:
        pose_timestamps (list[int]): Timestamps of supplied poses. Must be in ascending order.
        abs_poses (list[numpy.matrixlib.defmatrix.matrix]): SE3 matrices representing poses at the timestamps specified.
        requested_timestamps (list[int]): Timestamps for which interpolated timestamps are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    Raises:
        ValueError: if pose_timestamps and abs_poses are not the same length
        ValueError: if pose_timestamps is not in ascending order

    """
    start = time.time()
    requested_timestamps = requested_timestamps.copy()
    requested_timestamps.insert(0, origin_timestamp)
    requested_timestamps = np.array(requested_timestamps)
    pose_timestamps = np.array(pose_timestamps)

    if len(pose_timestamps) != len(abs_poses):
        raise ValueError('Must supply same number of timestamps as poses')

    abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))
    for i, pose in enumerate(abs_poses):
        if i > 0 and pose_timestamps[i-1] >= pose_timestamps[i]:
            raise ValueError('Pose timestamps must be in ascending order')
        abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3])
        abs_positions[:, i] = np.ravel(pose[0:3, 3])
    upper_indices = [bisect.bisect(pose_timestamps, pt) for pt in requested_timestamps]
    lower_indices = [u - 1 for u in upper_indices]

    if max(upper_indices) >= len(pose_timestamps):
        upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices]
    
    
    fractions = (requested_timestamps - pose_timestamps[lower_indices]) // \
                (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]

    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)
    sin_interp_indices = np.nonzero(d_array < 1)

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))

    scale0_array[sin_interp_indices] = \
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = \
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0)
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower \
                         + np.tile(scale1_array, (4, 1)) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]

    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    poses_mat = ml.zeros((4, 4 * len(requested_timestamps)))

    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])

    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])

    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1
    
    reference_pose = poses_mat[0:4, 0:4]
    poses_mat = np.linalg.solve(poses_mat[0:4, 0:4], poses_mat)

    poses_out = [0] * (len(requested_timestamps) - 1)
    for i in range(1, len(requested_timestamps)):
        poses_out[i - 1] = poses_mat[0:4, i * 4:(i + 1) * 4]
    end = time.time()
    
    return poses_out, reference_pose