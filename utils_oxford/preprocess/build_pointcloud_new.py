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
import sys
import os
import re
import numpy as np
import csv
import numpy.matlib as ml
import transform
from transform import build_se3_transform,get_transform_from_utm,build_se3_transform_batch, get_transform_from_utm_batch, build_xyzrpy_transform_batch
from interpolate_poses_new import interpolate_vo_poses, interpolate_ins_poses
from velodyne_new import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
import pyransac3d as pyrsc
from transform import *

Debug = False

def get_car_location_direction(timestamp, whichset=None):
    _, poses_current_time, _, _ = interpolate_ins_poses(None, [timestamp], timestamp,
                                                  use_rtk=False)
    poses_current_time = np.asarray(poses_current_time)
    location = poses_current_time[0:3, 3]
    direction = poses_current_time[0:3, 0:3]
    return location, direction
def get_related_transform(start_timestamp , end_timestamp): #### it return the norm and yaw
    pose_translation_start, pose_rotation_start = get_car_location_direction(start_timestamp)
    map_utm2pose_start = get_transform_from_utm(pose_rotation_start, pose_translation_start)
    location, direction = get_car_location_direction(end_timestamp)
    poses_end_time = get_transform_to_utm(direction, location)
    current2start = np.matmul(map_utm2pose_start, poses_end_time)
    return current2start

lidarsaved_dir = ''

def load_pointcloud_new_stamps(lidar_dir, poses_file, extrinsics_dir, timestamps, origin_time=-1):
    lidars = []
    for lidartime_i in timestamps:
        file = lidarsaved_dir + str(lidartime_i)+'.npy'
        transform_ = get_related_transform(origin_time , lidartime_i)
        lidar_i = np.load(file)
        lidar_i = np.matmul(transform_, lidar_i)
        lidars.append(lidar_i)
    
    pointcloud = np.concatenate(lidars, axis=1)
    
    return pointcloud, None


def build_pointcloud_new_stamps(lidar_dir, poses_file, extrinsics_dir, timestamps, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    if Debug:
        print(timestamps)

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = 'ins'

    with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                             G_posesource_laser)

    poses, poses_origin_time, ins_timestamps, abs_poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=False)#(poses_type == 'rtk'))

    pointcloud = np.empty((0,4))
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))
        
    use_rtk=False

    if Debug:
        print(timestamps)

    if Debug:
        print(timestamps)
        print("======")
    point_undis = []
    start_timestamps = 0
    if Debug:
        print('map integrating number:',len(timestamps))
        print(poses)
    for i in range(0, len(poses)):
        start_timestamps = 0
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')

        if not os.path.isfile(scan_path):
            continue

        ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
        ptcld,valid = velodyne_raw_to_pointcloud(ranges, intensities, angles)
        valid = valid.reshape(-1)
        ptcld = ptcld[...,valid]
        approximate_timestamps = approximate_timestamps[0]
        scan = ptcld
        scan = scan.transpose() 

        
        approximate_timestamps_expand = np.linspace(approximate_timestamps[0:-1], approximate_timestamps[1:], 33)[1:,...]
        approximate_timestamps_expand = approximate_timestamps_expand.reshape(-1, order='F')

        intve_temp = approximate_timestamps_expand[1]-approximate_timestamps_expand[0]
        intve_arr = np.arange(-32,0,1)*intve_temp+approximate_timestamps_expand[0]
        approximate_timestamps_expand= np.hstack((intve_arr, approximate_timestamps_expand))  

    
        approximate_timestamps_expand = approximate_timestamps_expand[valid]     
        
        scan_front_valid = scan[:,0]<-3

        plane1 = pyrsc.Plane()
        best_eq, best_inliers = plane1.fit(scan[:,0:3], 0.15)
        best_outliers = np.full((len(scan),), True)
        best_outliers[best_inliers] = False
        
        scan_front_valid = np.logical_and(scan_front_valid, best_outliers)
        
        
        
        scan = scan[scan_front_valid,...]
        approximate_timestamps_expand = approximate_timestamps_expand[scan_front_valid,...]
    
        reflectance = np.concatenate((reflectance, scan[:,3]))

        t1_ind = np.sum(ins_timestamps.reshape(1,-1) <= approximate_timestamps_expand.reshape(-1,1), axis=1)
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
        t_x1 = approximate_timestamps_expand - ins_timestamps[t1_ind]

        x = point_1_x + (t_x1/T_21)*(point_2_x - point_1_x)
        y = point_1_y + (t_x1/T_21)*(point_2_y - point_1_y)
        z = point_1_z + (t_x1/T_21)*(point_2_z - point_1_z)
        roll = point_1_roll + (t_x1/T_21)*(point_2_roll - point_1_roll)
        pitch = point_1_pitch + (t_x1/T_21)*(point_2_pitch - point_1_pitch)
        yaw = point_1_yaw + (t_x1/T_21)*(point_2_yaw - point_1_yaw)
        
        
        
        gps_vehicle = build_se3_transform_batch(np.stack([x,y,z,roll,pitch,yaw], axis=1))

        gps2_origin_time = get_transform_from_utm(poses_origin_time[0:3,0:3], poses_origin_time[0:3, 3].reshape(-1))
        pose_i = np.einsum('...jk,...kh->...jh', gps2_origin_time, gps_vehicle)##related to the selected frame at origin_time
        
        pose_i = np.einsum('...jk,...kh->...jh', pose_i, G_posesource_laser)
        points_hom = np.hstack((scan[:,0:3], np.ones((scan.shape[0],1)))).reshape(-1,4)
        point_undis = np.einsum('...jk,...k->...j', pose_i, points_hom)
        
        
        
        
        pointcloud = np.vstack((pointcloud, point_undis))
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud.transpose(), reflectance

def build_pointcloud_new(lidar_dir, poses_file, extrinsics_dir, start_time, end_time, origin_time=-1):
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")
    if Debug:
        print(timestamps)
    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = 'ins'

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses, poses_origin_time, ins_timestamps, abs_poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=False)
    else:
        pass

    pointcloud = np.empty((0,4))
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))
        
    use_rtk=False

    point_undis = []
    start_timestamps = 0
    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
        if not os.path.isfile(scan_path):
            continue
        ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
        ptcld,valid = velodyne_raw_to_pointcloud(ranges, intensities, angles)
        valid = valid.reshape(-1)
        ptcld = ptcld[...,valid]
        approximate_timestamps = approximate_timestamps[0]
        scan = ptcld
        scan = scan.transpose() 
        
        if start_timestamps == 0:
            start_timestamps = approximate_timestamps[-1]
            approximate_timestamps_expand = np.linspace(approximate_timestamps[0:-1], approximate_timestamps[1:], 33)[1:,...]
            approximate_timestamps_expand = approximate_timestamps_expand.reshape(-1, order='F')

            intve_temp = approximate_timestamps_expand[1]-approximate_timestamps_expand[0]
            intve_arr = np.arange(-32,0,1)*intve_temp+approximate_timestamps_expand[0]
            approximate_timestamps_expand= np.hstack((intve_arr, approximate_timestamps_expand))  
                
        else:
            approximate_timestamps = np.insert(approximate_timestamps, 0, start_timestamps)  
            approximate_timestamps_expand = np.linspace(approximate_timestamps[0:-1], approximate_timestamps[1:], 33)[1:,...]
            approximate_timestamps_expand = approximate_timestamps_expand.reshape(-1, order='F')
            start_timestamps = approximate_timestamps[-1]    
                
        approximate_timestamps_expand = approximate_timestamps_expand[valid]       
        
        scan_front_valid = scan[:,0]<-3

        scan = scan[scan_front_valid,...]
        approximate_timestamps_expand = approximate_timestamps_expand[scan_front_valid,...]
    
        reflectance = np.concatenate((reflectance, scan[:,3]))
        
        
        t1_ind = np.sum(ins_timestamps.reshape(1,-1) <= approximate_timestamps_expand.reshape(-1,1), axis=1)
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
        t_x1 = approximate_timestamps_expand - ins_timestamps[t1_ind]

        x = point_1_x + (t_x1/T_21)*(point_2_x - point_1_x)
        y = point_1_y + (t_x1/T_21)*(point_2_y - point_1_y)
        z = point_1_z + (t_x1/T_21)*(point_2_z - point_1_z)
        roll = point_1_roll + (t_x1/T_21)*(point_2_roll - point_1_roll)
        pitch = point_1_pitch + (t_x1/T_21)*(point_2_pitch - point_1_pitch)
        yaw = point_1_yaw + (t_x1/T_21)*(point_2_yaw - point_1_yaw)
        
        
        
        gps_vehicle = build_se3_transform_batch(np.stack([x,y,z,roll,pitch,yaw], axis=1))

        gps2_origin_time = get_transform_from_utm(poses_origin_time[0:3,0:3], poses_origin_time[0:3, 3].reshape(-1))
        pose_i = np.einsum('...jk,...kh->...jh', gps2_origin_time, gps_vehicle)##related to the selected frame at origin_time
        
        pose_i = np.einsum('...jk,...kh->...jh', pose_i, G_posesource_laser)
        points_hom = np.hstack((scan[:,0:3], np.ones((scan.shape[0],1)))).reshape(-1,4)
        point_undis = np.einsum('...jk,...k->...j', pose_i, points_hom)

        pointcloud = np.vstack((pointcloud, point_undis))
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud.transpose(), reflectance


def get_angle(x, y):
    if (x > 0) and (y > 0):
        angle = np.degrees(np.arctan(np.absolute(y/x)))
    elif (x < 0) and (y > 0):
        angle = 180 - np.degrees(np.arctan(np.absolute(y/x)))
    elif (x < 0) and (y < 0):
        angle = 180 + np.degrees(np.arctan(np.absolute(y/x)))
    elif (x > 0) and (y < 0):
        angle = 360 - np.degrees(np.arctan(np.absolute(y/x)))
    elif (y == 0) and (x > 0):
        angle = 0.0
    elif (y == 0) and (x < 0):
        angle = 180.0
    elif (y < 0) and (x == 0):
        angle = 270.0
    elif (y > 0) and (x == 0):
        angle = 90.0
    return angle



if __name__ == "__main__":
    import argparse
    import open3d

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')

    args = parser.parse_args()

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    timestamps_path = os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    with open(timestamps_path) as timestamps_file:
        start_time = int(next(timestamps_file).split(' ')[0])

    end_time = start_time + 2e7

    pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file,
                                               args.extrinsics_dir, start_time, end_time)

    if reflectance is not None:
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
    else:
        colours = 'gray'

    # Pointcloud Visualisation using Open3D
    vis = open3d.Visualizer()
    vis.create_window(window_name=os.path.basename(__file__))
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
    render_option.point_color_option = open3d.PointColorOption.ZCoordinate
    coordinate_frame = open3d.geometry.create_mesh_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64)))
    pcd.colors = open3d.utility.Vector3dVector(np.tile(colours[:, np.newaxis], (1, 3)).astype(np.float64))
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()