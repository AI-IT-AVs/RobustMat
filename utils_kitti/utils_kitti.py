import utm
import numpy as np
import math
import networkx as nx
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform
import os
import matplotlib.pylab as pt
import global_config
import torch
from torchvision import transforms
import torchvision.transforms.functional as tranF
import scipy as sp
import scipy.sparse
import cv2
from torch.utils.data import Dataset, DataLoader
import pykitti
from imgaug import augmenters as iaa
import random

random_seed = 1023
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
###
seq = iaa.Sequential([
    iaa.imgcorruptlike.Spatter(severity=3),
    iaa.imgcorruptlike.Spatter(severity=1),
    iaa.imgcorruptlike.GaussianNoise(severity=3)
    iaa.imgcorruptlike.Brightness(severity=2),  
    iaa.imgcorruptlike.Saturate(severity=3),  
])


# Change this to the directory where you store KITTI data
basedir = ''

len_seq = np.asarray(global_config.len_seq)
len_seq_cum = np.cumsum(len_seq)
len_seq_cum_temp = np.concatenate(([0], len_seq_cum))
kitti_seq = []
for seq_i in range(11):
    frames = range(0, global_config.len_seq[seq_i], 1)
    sequence = '{0:02d}'.format(seq_i)
    dataset_i = pykitti.odometry(basedir, sequence, frames=frames)
    kitti_seq.append(dataset_i)

glo_seg_class = []


def get_graph_per_frame(idx_rec_pair, savedpath='', shift=None):
    ImagePatchBackground = True
    dataset_idx = np.searchsorted(len_seq_cum, [idx_rec_pair[0]], side='right')[0] ####determine which sequence
    dataset = kitti_seq[dataset_idx]
    idx_frame = idx_rec_pair[0] - len_seq_cum_temp[dataset_idx]  #####determine the index in the sequence
    image_curr = np.array(dataset.get_cam3(idx_frame))
    num_rows = image_curr.shape[0]
    num_cols = image_curr.shape[1]
    image_curr = seq.augment_images([image_curr.astype(np.uint8)])
    image_curr = image_curr[0]
    idx_frame_next = idx_rec_pair[1] - len_seq_cum_temp[dataset_idx] 
    image_next = np.array(dataset.get_cam3(idx_frame_next))
    num_rows_next = image_next.shape[0]
    num_cols_next = image_next.shape[1]
    image_next = seq.augment_images([image_next.astype(np.uint8)])
    image_next = image_next[0]
    Value_padding = 15
    minimum_pixel_up = 10  
    minimum_pixel_low = 10  
    maximum_pixel_up = 10000  
    maximum_pixel_low = 10000
    minimum_pixel = minimum_pixel_up
    maximum_pixel = maximum_pixel_low
    Thre_frame_i_object = 40
    Thre_pixel_coor_len = 500

    minimum_pixel_road = 200
    saved_dir = savedpath
    num_valid_lidar_points = 5 
    Threshold_valid_location_utm = 0.4 
    Threshold_seg_ref_point = 4 
    Threshold_seg_ref_point_2 = 1 
    num_fixed = 16  

    idx_rec = idx_rec_pair[0]
    valid_location_utm_frame0 = []
    valid_location_frame_0 = []
    seg_location_utm_frame_0 = []
    x = []
    frame_i_objects_inrange_0, frame_i_objects_outofrange_0 = np.load(os.path.join(saved_dir, str(idx_rec) + '.npy'),allow_pickle=True)
    if len(frame_i_objects_inrange_0) < Thre_frame_i_object:
        maximum_pixel = maximum_pixel_up
        minimum_pixel = minimum_pixel_low
    node_idx_0 = 0
    lidar_idx_0 = 0
    G_0 = nx.Graph()
    list_nodes_0 = []
    image_segs_0 = []
    seg_feature_0 = []
    seg_class_indx_0 = []
    seg_lidaradj_0 = []
    seg_dis2vehicle_0 = []
    seg_lidarfeature_0 = []
    list_nodes_object_i_location_0 = np.empty((0, 3))
    car_location_0 = frame_i_objects_inrange_0[0]['car_location']
    car_direction_0 = frame_i_objects_inrange_0[0]['car_direction']
    edges_0 = []
    seg_pixel_coors_0 = []
    pixel_coors_len_0 = []
    
    if shift is not None:
        shift1 = np.random.uniform(-0.2, 0.2, 3)
        car_location_0[0] = car_location_0[0] + shift1[0]
        car_location_0[1] = car_location_0[1] + shift1[1]

    image_size = 256
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    for object_i in frame_i_objects_inrange_0:
        if len(object_i['pixel_coors']) > minimum_pixel and len(object_i['pixel_coors']) < maximum_pixel:
            if ImagePatchBackground:
                tempcolorind = np.unravel_index(object_i['pixel_coors'], (num_rows, num_cols), order='F')
                up = np.min(tempcolorind[0])
                down = np.max(tempcolorind[0])
                left = np.min(tempcolorind[1])
                right = np.max(tempcolorind[1])
                seg_pixel_coors = np.array([[up, left], 
                                            [up, right],
                                            [down, left],
                                            [down, right]]).astype(np.float32)
            if ImagePatchBackground:
                height_padding = Value_padding
                width_padding = Value_padding
                if (left-width_padding>=0 and right+width_padding<=num_cols) and (up-height_padding>=0 and down+height_padding<=num_rows):
                    ori_seg = image_curr[up-height_padding:down+height_padding+1,left-width_padding:right+width_padding+1,:]/255
                else:
                    ori_seg = image_curr[up:down+1,left:right+1,:]/255
            else: 
                ori_seg = object_i['image_seg']
            class_indx = object_i['seg_class']

            if class_indx == global_config.pole_class or class_indx == global_config.traffic_light_class or class_indx == global_config.traffic_sign_class:
                pass
            else:
                continue

            if class_indx == global_config.road_class:
                lidar_points_utm = object_i['lidar_points']

                ###### transform lidar points from global to carview #################################
                points_hom = np.ones((lidar_points_utm.shape[0], 4))
                points_hom[:, 0:3] = lidar_points_utm
                transvehicle_view = get_transform_from_utm(object_i['car_direction'], object_i['car_location'])
                trans_to_center = transvehicle_view
                lidar_points = (np.matmul(trans_to_center, points_hom.T)).T[:, 0:3]

                valid_lidar_points = lidar_points
                if object_i['left_right'] == 0:
                    valid_location_utm = np.asarray([-10000.0, 0, 1])
                else:
                    valid_location_utm = np.asarray([10000.0, 0, 1])

                ######anchor location (in car coordinate system)for one seg#############
                locationindex = np.argmin(np.abs(valid_lidar_points[:, 0]))
                location__ = valid_lidar_points[locationindex, :]

                ####### filter according to number of lidar points#####################
                if len(valid_lidar_points) < num_valid_lidar_points:
                    continue
                if len(object_i['pixel_coors']) < minimum_pixel_road:
                    continue
                lidar_coorstemp = object_i['lidar_coors']

            else:
                lidar_points_utm = object_i['lidar_points']

                ###### transform lidar points from global to carview #################################
                points_hom = np.ones((lidar_points_utm.shape[0], 4))
                points_hom[:, 0:3] = lidar_points_utm
                transvehicle_view = get_transform_from_utm(object_i['car_direction'], object_i['car_location'])
                trans_to_center = transvehicle_view
                lidar_points = (np.matmul(trans_to_center, points_hom.T)).T[:, 0:3]
				
                lidarcoorsttt = np.unravel_index(object_i['lidar_coors'], (num_rows, num_cols), order='F')
                up = np.min(lidarcoorsttt[0])
                down = np.max(lidarcoorsttt[0])
                left = np.min(lidarcoorsttt[1])
                right = np.max(lidarcoorsttt[1])
                aa = (up + down) / 2.0
                bb = (left + right) / 2.0
                difff = np.linalg.norm(np.array([aa, bb]) - np.stack((lidarcoorsttt[0], lidarcoorsttt[1]), axis=1),axis=1)
                seg_ref_point_ind = np.argmin(difff)


                ####filter lidar points according to the distance from the seg_ref_point_ind, need to adjust
                if class_indx == global_config.windows_class:
                    sel_pts = np.logical_and(
                        np.logical_and(np.abs(lidar_points[:, 1] - lidar_points[seg_ref_point_ind, 1]) < 3,
                                       np.abs(lidar_points[:, 0] - lidar_points[seg_ref_point_ind, 0]) < 3),
                        np.abs(lidar_points[:, 2] - lidar_points[seg_ref_point_ind, 2]) < 3)
                else:
                    sel_pts = np.logical_and(
                        np.logical_and(np.abs(lidar_points[:, 1] - lidar_points[seg_ref_point_ind, 1]) < Threshold_seg_ref_point,
                                       np.abs(lidar_points[:, 0] - lidar_points[seg_ref_point_ind, 0]) < Threshold_seg_ref_point),
                        np.abs(lidar_points[:, 2] - lidar_points[seg_ref_point_ind, 2]) < Threshold_seg_ref_point_2)

                valid_lidar_points = lidar_points[sel_pts, :]
                valid_location_utm = np.average(lidar_points_utm[sel_pts, :], axis=0)


                ####anchor location (in car coordinate system)for one seg#############
                locationindex = np.argmin(np.abs(valid_lidar_points[:, 0]))
                location__ = valid_lidar_points[locationindex, :]


                ######### filter according to number of lidar points#####################
                if len(valid_lidar_points) < num_valid_lidar_points:
                    continue
                lidar_coorstemp = object_i['lidar_coors'][sel_pts]
            lidar_points_global_idx = list(range(lidar_idx_0, lidar_idx_0 + len(lidar_points)))
            seg_lidaradj_0.append(torch.empty(0, 1))
            seg_dis2vehicle_0.append(torch.empty(0, 1))

            w = ori_seg.shape[1]
            h = ori_seg.shape[0]
            if h > w:
                padd = math.ceil((h - w) / 2.0)
                ori_seg = np.concatenate((np.tile(global_config.image_mean, (h, padd, 1)), ori_seg,
                                          np.tile(global_config.image_mean, (h, padd, 1))), axis=1)
            else:
                padd = math.ceil((w - h) / 2.0)
                ori_seg = np.concatenate((np.tile(global_config.image_mean, (padd, w, 1)), ori_seg,
                                          np.tile(global_config.image_mean, (padd, w, 1))), axis=0)
            ori_seg = ori_seg.astype(np.float32)
            ori_seg = torch.from_numpy(np.transpose(ori_seg, (2, 0, 1)))
            seg = transform(ori_seg)

            image_segs_0.append(seg)
            seg_pixel_coors_0.append(seg_pixel_coors)

            pixel_coors_len = len(object_i['pixel_coors'])
            pixel_coors_len_0.append(pixel_coors_len)

            node_i = {'lidar_points': lidar_points, 'lidar_points_utm': lidar_points_utm,
                      'lidar_points_global_idx': np.asarray(lidar_points_global_idx),
                      'location_utm': valid_location_utm, 'valid_lidar_points': valid_lidar_points,
                      'lidar_coors': lidar_coorstemp} 
            list_nodes_0.append(node_i)

            seg_feature_0.append(global_config.class_feature[class_indx])
            seg_class_indx_0.append(class_indx)
            seg_lidarfeature_0.append(torch.from_numpy(np.asarray(valid_lidar_points).astype(np.float32)))

            valid_location_utm_frame0.append(valid_location_utm)
            valid_location_frame_0.append(location__) 

            node_idx_0 += 1
            lidar_idx_0 += lidar_idx_0 + len(lidar_points)  ###not used

    num_inrange_0 = node_idx_0

    if num_inrange_0 < 5:
        return None

    image_segs_0 = torch.cat(image_segs_0, dim=0).view(-1, 3, image_size, image_size)
    seg_feature_0 = torch.from_numpy(np.asarray(seg_feature_0).astype(np.float32))
    seg_class_indx_0 = torch.from_numpy(np.asarray(seg_class_indx_0).astype(np.float32))
    valid_location_frame_0 = torch.from_numpy(np.asarray(valid_location_frame_0).astype(np.float32))

    car_location_0 = torch.from_numpy(car_location_0.astype(np.float32))
    car_direction_0 = torch.from_numpy(car_direction_0.astype(np.float32))

    valid_location_utm_frame0 = np.asarray(valid_location_utm_frame0)

    check = 0
    matched_num = 0
    idx_rec = idx_rec_pair[1]
    matchedseg = []
    x = []
    frame_i_objects_inrange, frame_i_objects_outofrange = np.load(os.path.join(saved_dir, str(idx_rec) + '.npy'),allow_pickle=True)
    if len(frame_i_objects_inrange) < Thre_frame_i_object:
        maximum_pixel = maximum_pixel_up
        minimum_pixel = minimum_pixel_low
    node_idx = 0
    lidar_idx = 0
    G = nx.Graph()
    list_nodes = []
    image_segs = []
    seg_feature = []
    seg_class_indx = []
    seg_lidaradj = []
    seg_dis2vehicle = []
    valid_location_frame = []
    seg_location_utm_frame = []
    list_nodes_object_i_location = np.empty((0, 3))
    seg_lidarfeature = []
    car_location = frame_i_objects_inrange[0]['car_location']
    car_direction = frame_i_objects_inrange[0]['car_direction']
    seg_pixel_coors_1 = []
    if shift is not None:
        shift1 = np.random.uniform(-0.2, 0.2, 3)
        car_location[0] = car_location[0] + shift1[0]
        car_location[1] = car_location[1] + shift1[1]

    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    for object_i in frame_i_objects_inrange:

        if len(object_i['pixel_coors']) > minimum_pixel and len(object_i['pixel_coors']) < maximum_pixel:
            if ImagePatchBackground:
                tempcolorind = np.unravel_index(object_i['pixel_coors'], (num_rows_next, num_cols_next), order='F')
                up = np.min(tempcolorind[0])
                down = np.max(tempcolorind[0])
                left = np.min(tempcolorind[1])
                right = np.max(tempcolorind[1])
                seg_pixel_coors = np.array([[up, left], 
                                            [up, right],
                                            [down, left],
                                            [down, right]]).astype(np.float32)
                
            if ImagePatchBackground:
                height_padding = Value_padding
                width_padding = Value_padding
                if (left-width_padding>=0 and right+width_padding<=num_cols_next) and (up-height_padding>=0 and down+height_padding<=num_rows_next):
                    ori_seg = image_curr[up-height_padding:down+height_padding+1,left-width_padding:right+width_padding+1,:]/255
                else:
                    ori_seg = image_curr[up:down+1,left:right+1,:]/255
            else: 
                ori_seg = object_i['image_seg']
            class_indx = object_i['seg_class']
            if class_indx == global_config.pole_class or class_indx == global_config.traffic_light_class or class_indx == global_config.traffic_sign_class:
                pass
            else:
                continue
            if class_indx == global_config.road_class:
                lidar_points_utm = object_i['lidar_points']

                #####transform lidar points from global to carview #################################
                points_hom = np.ones((lidar_points_utm.shape[0], 4))
                points_hom[:, 0:3] = lidar_points_utm
                transvehicle_view = get_transform_from_utm(object_i['car_direction'], object_i['car_location'])
                trans_to_center = transvehicle_view
                lidar_points = (np.matmul(trans_to_center, points_hom.T)).T[:, 0:3]

                valid_lidar_points = lidar_points

                locationindex = np.argmin(np.abs(valid_lidar_points[:, 0]))
                location__ = valid_lidar_points[locationindex, :]


                ############filter according to number of lidar points#####################
                if len(valid_lidar_points) < num_valid_lidar_points:
                    continue
                if len(object_i['pixel_coors']) < minimum_pixel_road:
                    continue
                lidar_coorstemp = object_i['lidar_coors']

            else:
                lidar_points_utm = object_i['lidar_points']

                ####### transform lidar points from global to carview #################################
                points_hom = np.ones((lidar_points_utm.shape[0], 4))
                points_hom[:, 0:3] = lidar_points_utm
                transvehicle_view = get_transform_from_utm(object_i['car_direction'], object_i['car_location'])
                trans_to_center = transvehicle_view
                lidar_points = (np.matmul(trans_to_center, points_hom.T)).T[:, 0:3]

                lidarcoorsttt = np.unravel_index(object_i['lidar_coors'], (num_rows, num_cols), order='F')
                up = np.min(lidarcoorsttt[0])
                down = np.max(lidarcoorsttt[0])
                left = np.min(lidarcoorsttt[1])
                right = np.max(lidarcoorsttt[1])
                aa = (up + down) / 2.0
                bb = (left + right) / 2.0
                difff = np.linalg.norm(np.array([aa, bb]) - np.stack((lidarcoorsttt[0], lidarcoorsttt[1]), axis=1),
                                       axis=1)
                seg_ref_point_ind = np.argmin(difff)


                ####filter lidar points according to the distance from the seg_ref_point_ind, need to adjust
                if class_indx == global_config.windows_class:
                    sel_pts = np.logical_and(
                        np.logical_and(np.abs(lidar_points[:, 1] - lidar_points[seg_ref_point_ind, 1]) < 3,
                                       np.abs(lidar_points[:, 0] - lidar_points[seg_ref_point_ind, 0]) < 3),
                        np.abs(lidar_points[:, 2] - lidar_points[seg_ref_point_ind, 2]) < 3)
                else:
                    sel_pts = np.logical_and(
                        np.logical_and(np.abs(lidar_points[:, 1] - lidar_points[seg_ref_point_ind, 1]) < Threshold_seg_ref_point,
                                       np.abs(lidar_points[:, 0] - lidar_points[seg_ref_point_ind, 0]) < Threshold_seg_ref_point),
                        np.abs(lidar_points[:, 2] - lidar_points[seg_ref_point_ind, 2]) < Threshold_seg_ref_point_2)

                valid_lidar_points = lidar_points[sel_pts, :]
                valid_location_utm = np.average(lidar_points_utm[sel_pts, :], axis=0)
				
                #### valid_location_utm is for matching the segs between two frames#####
                locationindex = np.argmin(np.abs(valid_lidar_points[:, 0]))
                location__ = valid_lidar_points[locationindex, :]
                lidar_coorstemp = object_i['lidar_coors'][sel_pts]

            ############################ matching segs #####################
            if class_indx == global_config.road_class:
                if len(valid_lidar_points) < num_valid_lidar_points:
                    continue
                if object_i['left_right'] == 0:
                    matched = np.where(valid_location_utm_frame0[:, 0] < -9999)[0]
                else:
                    matched = np.where(valid_location_utm_frame0[:, 0] > 9999)[0]
                if len(matched) == 0 or len(matched) > 1:
                    continue
            else:
                if len(valid_lidar_points) < num_valid_lidar_points:
                    continue
                
                matched = np.where(np.sum(np.abs(valid_location_utm - valid_location_utm_frame0),
                                          axis=1) < Threshold_valid_location_utm)[0]


            if len(matched) > 1:
                continue
            if len(matched) == 0 or (abs(len(object_i['pixel_coors']) - pixel_coors_len_0[
                matched[0]]) > Thre_pixel_coor_len and class_indx == global_config.windows_class):
                matchedseg.append(-100)  
            else:
                matchedseg.append(matched[0])
                matched_num = matched_num + 1

            valid_lidar_points = valid_lidar_points[valid_lidar_points[:, 1].argsort()]

            
            lidar_points_global_idx = list(range(lidar_idx_0, lidar_idx_0 + len(lidar_points)))
            seg_lidaradj.append(torch.empty(0, 1))
            seg_dis2vehicle.append(torch.empty(0, 1))

            w = ori_seg.shape[1]
            h = ori_seg.shape[0]
            if h > w:
                padd = math.ceil((h - w) / 2.0)
                ori_seg = np.concatenate((np.tile(global_config.image_mean, (h, padd, 1)), ori_seg,
                                          np.tile(global_config.image_mean, (h, padd, 1))), axis=1)
            else:
                padd = math.ceil((w - h) / 2.0)
                ori_seg = np.concatenate((np.tile(global_config.image_mean, (padd, w, 1)), ori_seg,
                                          np.tile(global_config.image_mean, (padd, w, 1))), axis=0)
            ori_seg = ori_seg.astype(np.float32)
            ori_seg = torch.from_numpy(np.transpose(ori_seg, (2, 0, 1)))
            seg = transform(ori_seg)

            image_segs.append(seg)
            seg_pixel_coors_1.append(seg_pixel_coors)

            node_i = {'lidar_points': lidar_points, 'lidar_points_utm': lidar_points_utm,
                      'lidar_points_global_idx': np.asarray(lidar_points_global_idx),
                      'location_utm': valid_location_utm, 'valid_lidar_points': valid_lidar_points,
                      'lidar_coors': lidar_coorstemp}  #####node_i not used in this version
            list_nodes_0.append(node_i)

            seg_feature.append(global_config.class_feature[class_indx])
            seg_class_indx.append(class_indx)
            seg_lidarfeature.append(torch.from_numpy(np.asarray(valid_lidar_points).astype(np.float32)))

            valid_location_frame.append(location__)  

            node_idx += 1
            lidar_idx += lidar_idx + len(lidar_points)

    num_inrange = node_idx
    if num_inrange < 3:
        return None
    image_segs = torch.cat(image_segs, dim=0).view(-1, 3, image_size, image_size)
    seg_feature = torch.from_numpy(np.asarray(seg_feature).astype(np.float32))
    seg_class_indx = torch.from_numpy(np.asarray(seg_class_indx).astype(np.float32))
    valid_location_frame = torch.from_numpy(np.asarray(valid_location_frame).astype(np.float32))

        
    car_location = torch.from_numpy(car_location.astype(np.float32))
    car_direction = torch.from_numpy(car_direction.astype(np.float32))
    
    ############## delete repeated matched segs ###################################
    matched_unique = [ i for i in set(matchedseg) if i!= -100]
    if len(matched_unique) < matched_num:
        matchedseg = np.array(matchedseg)
        match_repeat = [j for j in matched_unique if len(np.where(matchedseg ==j)[0])>1]
        for j in match_repeat:
            matchedseg[np.where(np.array(matchedseg) ==j)[0]] = -200
        matchseginframe1 = np.where(matchedseg!=-200)[0]
        matchedseg = matchedseg[matchseginframe1]
        matchedseg = matchedseg.tolist()
      
    
    if matched_num<3 or len(matched_unique)<3:
        return None
        
    if len(matched_unique) < matched_num:
        image_segs = image_segs[matchseginframe1,...]
        seg_feature = seg_feature[matchseginframe1,...]
        seg_class_indx = seg_class_indx[matchseginframe1,...]
        valid_location_frame = valid_location_frame[matchseginframe1,...]
        seg_lidarfeature = [seg_lidarfeature[ii] for ii in matchseginframe1]
        seg_lidaradj = [seg_lidaradj[ii] for ii in matchseginframe1]
        seg_dis2vehicle = [seg_dis2vehicle[ii] for ii in matchseginframe1]
        seg_pixel_coors_1 = [seg_pixel_coors_1[ii] for ii in matchseginframe1]
    adj_full = torch.from_numpy(np.asarray(normalize_adj(np.ones((len(matchedseg),len(matchedseg))))).astype(np.float32))
    
    
    matchedseg = torch.IntTensor(matchedseg)
    x.append((image_segs_0, car_location_0, car_direction_0, seg_feature_0, seg_class_indx_0, seg_lidaradj_0, seg_dis2vehicle_0, seg_pixel_coors_0, valid_location_frame_0, adj_full, matchedseg))
    x.append((image_segs, car_location, car_direction, seg_feature, seg_class_indx, seg_lidaradj, seg_dis2vehicle, seg_pixel_coors_1,valid_location_frame, adj_full, matchedseg)) 
    
    return x
    
    
def get_transform_to_utm(direction, location):
    transform_to_utm = np.eye(4)
    transform_to_utm[0:3, 0:3] = direction
    transform_to_utm[0:3, 3] = location

    return transform_to_utm


def get_transform_from_utm(direction, location):  ##from utm to vehicle_view
    transform_to_utm = get_transform_to_utm(direction, location)
    trans = np.eye(4)
    rot = np.transpose(transform_to_utm[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_utm[0:3, 3])
    return trans



def extract_semantic_file_name_from_image_file_name(file_name_image):
    seq_name = os.path.basename(file_name_image)
    seq_name = seq_name.split('.')[0]

    return seq_name


def get_car_location_direction(timestamp, dataset_idx):
    dataset = kitti_seq[dataset_idx]
    pose = dataset.poses[timestamp]
    location = pose[0:3,3]
    direction = pose[0:3,0:3]
    return location, direction 


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / math.pi


def get_mean_std(current_date_dir, savedpath=''):  # idx_rec is the indice of the image in the whole image dir
    image_means = []
    image_stds = []
    iii = 0
    for idx_rec in range(942):
        frame_i_objects_inrange, frame_i_objects_outofrange = np.load(
            os.path.join(savedpath, current_date_dir, str(idx_rec) + '.npy'), allow_pickle=True)
        for object_i in frame_i_objects_inrange:
            inxx = object_i['image_seg'].reshape(-1, 3)
            image_means.append(np.mean(inxx, axis=0))
            image_stds.append(np.std(inxx, axis=0))
        for object_i in frame_i_objects_inrange:
            inxx = object_i['image_seg'].reshape(-1, 3)
            image_means.append(np.mean(inxx, axis=0))
            image_stds.append(np.std(inxx, axis=0))

    mean = np.mean(np.asarray(image_means), axis=0)
    std = np.mean(np.asarray(image_stds), axis=0)
    print(np.asarray(image_means).shape)
    print(np.asarray(image_stds).shape)
    print('mean: {:s}'.format(str(mean)))
    print('std: {:s}'.format(str(std)))

    return image_means, image_stds


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = scipy.sparse.diags(r_inv_sqrt).todense()
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


class LandmarksDataset_train(Dataset):
    def __init__(self, savedpath='', shift=False):
        self.savedpath = savedpath
        self.valid_idx = np.load('*.npy', allow_pickle=True) 
    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        iidx = idx
        x = []
        while 1:
            iidx_pair = [int(self.valid_idx[iidx, 0]), int(self.valid_idx[iidx, 1])]
            x = get_graph_per_frame(iidx_pair, savedpath=self.savedpath, shift=None)
            if x is not None:
                break
            else:
                iidx = np.random.randint(len(self.valid_idx))
        return x


class LandmarksDataset_test(Dataset):

    def __init__(self, savedpath='', shift=False):
        self.savedpath = savedpath
        self.valid_idx = np.load('*.npy', allow_pickle=True) 

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        iidx = idx
        x = []
        while 1:
            iidx_pair = [int(self.valid_idx[iidx, 0]), int(self.valid_idx[iidx, 1])]
            x = get_graph_per_frame(iidx_pair, savedpath=self.savedpath, shift=None)
            if x is not None:
                break
            else:
                iidx = np.random.randint(len(self.valid_idx))
        return x                 
