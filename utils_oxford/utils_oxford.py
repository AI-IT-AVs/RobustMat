import utm
import sys
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
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from preprocess.image import load_image
from preprocess.build_pointcloud_new import build_pointcloud_new, build_pointcloud_new_stamps, load_pointcloud_new_stamps
from preprocess.transform import build_se3_transform, so3_to_euler, euler_to_so3, get_transform_from_utm, \
    get_transform_to_utm
from preprocess.camera_model_new import CameraModel
from preprocess.interpolate_poses_new import interpolate_vo_poses, interpolate_ins_poses
import pandas as pd
import re
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
import pyransac3d as pyrsc

glo_seg_class = []

with open(global_config.image_extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None
with open(os.path.join(global_config.extrinsics_dir, 'ins.txt')) as extrinsics_file:
    extrinsics = next(extrinsics_file)
    G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    G_camera_posesource = np.asarray(G_camera_posesource)

import matplotlib.colors as mcolors
colorlist = list(mcolors.CSS4_COLORS)

import random
import numpy as np
random_seed = 1023
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.imgcorruptlike.Spatter(severity=3),
    iaa.imgcorruptlike.Spatter(severity=1),
    iaa.imgcorruptlike.GaussianNoise(severity=3),
    iaa.imgcorruptlike.Brightness(severity=2),  
    iaa.imgcorruptlike.Saturate(severity=3),  
])

objects_saved_dir = ''

def get_graph_traning_testing(image_frame_time0, image_frame_time1):
    Value_padding = 0
    
    Debug = False
    
    if os.path.exists(objects_saved_dir + str(image_frame_time0) + ".pt"):
        objects_frame0 = torch.load(objects_saved_dir + str(image_frame_time0) + ".pt")
    else:
        return None
    if os.path.exists(objects_saved_dir + str(image_frame_time1) + ".pt"):
        objects_frame1 = torch.load(objects_saved_dir + str(image_frame_time1) + ".pt")
    else:
        return None

    images_frame0 = objects_frame0['image_segs']
    seg_lidarpoints_frame0 = objects_frame0['seg_lidarpoints']
    valid_location_utm_frame0 = objects_frame0['valid_location_utm_map']
    seg_class_indx_frame0 = objects_frame0['seg_class_indx']
    win_plane_ratio_frame0 = objects_frame0['win_plane_ratio']
    seg_image_coord_frame0 = objects_frame0['seg_image_coord']

    images_frame1 = objects_frame1['image_segs']
    seg_lidarpoints_frame1 = objects_frame1['seg_lidarpoints']
    valid_location_utm_frame1 = objects_frame1['valid_location_utm_map']
    seg_class_indx_frame1 = objects_frame1['seg_class_indx']
    win_plane_ratio_frame1 = objects_frame1['win_plane_ratio']
    seg_image_coord_frame1 = objects_frame1['seg_image_coord']

    if (images_frame0 is None) or (images_frame1 is None):
        return None
    
    matched_num = 0
    matchedseg_inframe0 = []
    matchedseg_inframe1 = []
    if seg_class_indx_frame0 is None:
        return None
    else:
        matchedseg_inframe0 = np.ones(len(seg_class_indx_frame0)).astype(int)*-100
        matchedseg_inframe1 = np.ones(len(seg_class_indx_frame0)).astype(int)*-100
    
    pole_indexs_frame0 = np.where(seg_class_indx_frame0 == global_config.pole_class)[0]#index under  full segmentations  
    pole_indexs_frame1 = np.where(seg_class_indx_frame1 == global_config.pole_class)[0]#index under  full segmentations  
    if len(pole_indexs_frame0)>0:
        
        for pole_i_frame0 in pole_indexs_frame0: ##pole_i_frame0 index under  full segmentations  
            distance_to_frame1_poles = LA.norm(valid_location_utm_frame0[pole_i_frame0,0:2] - valid_location_utm_frame1[pole_indexs_frame1,0:2], axis=1)
            nearby_poles = np.where(distance_to_frame1_poles<4)[0] #nearby_poles is index under pole_indexs_frame1
            if Debug:
                print('nearby poles A: ',nearby_poles)
            if len(nearby_poles)>0:
                
                if len(nearby_poles)==1:
                    if distance_to_frame1_poles[nearby_poles[0]] < 2.5:
                        matchedseg_inframe0[pole_i_frame0] = pole_i_frame0
                        matchedseg_inframe1[pole_i_frame0] = pole_indexs_frame1[nearby_poles[0]]
                else:
                    ### there exist some near multiple poles
                    nearby_poles = np.where(distance_to_frame1_poles<1.6)[0]  #nearby_poles is index under pole_indexs_frame1
                    if len(nearby_poles) == 1:
                        matchedseg_inframe0[pole_i_frame0] = pole_i_frame0
                        matchedseg_inframe1[pole_i_frame0] = pole_indexs_frame1[nearby_poles[0]]
                        
    ################# matching traffic sign #########################
    traffic_sign_indexs_frame0 = np.where(seg_class_indx_frame0 == global_config.traffic_sign_class)[0] #index under  full segmentations  
    traffic_sign_indexs_frame1 = np.where(seg_class_indx_frame1 == global_config.traffic_sign_class)[0] #index under  full segmentations  
    if len(traffic_sign_indexs_frame0)>0:
        
        for traffic_sign_i_frame0 in traffic_sign_indexs_frame0:##traffic_sign_i_frame0 index under  full segmentations  
            if (traffic_sign_i_frame0 is None) or (traffic_sign_indexs_frame1 is None):
                break
            if (valid_location_utm_frame0 is None) or (valid_location_utm_frame1 is None):
                break
            distance_to_frame1_traffic_signs_xy = LA.norm(valid_location_utm_frame0[traffic_sign_i_frame0,0:2] - valid_location_utm_frame1[traffic_sign_indexs_frame1,0:2], axis=1)
            distance_to_frame1_traffic_signs_z = np.abs(valid_location_utm_frame0[traffic_sign_i_frame0,2] - valid_location_utm_frame1[traffic_sign_indexs_frame1,2])
            distance_to_frame1_traffic_signs_xyz = np.logical_and(distance_to_frame1_traffic_signs_xy < 4,
                                                                 distance_to_frame1_traffic_signs_z < 1)
            nearby_traffic_signs = np.where(distance_to_frame1_traffic_signs_xyz)[0] ##nearby_traffic_signs under traffic_sign_indexs_frame1
            if Debug:
                print('nearby sign A: ',nearby_traffic_signs)
            if len(nearby_traffic_signs)>0:
                
                if len(nearby_traffic_signs)==1:
                    if distance_to_frame1_traffic_signs_xy[nearby_traffic_signs[0]] < 2.2:
                        matchedseg_inframe0[traffic_sign_i_frame0] = traffic_sign_i_frame0
                        matchedseg_inframe1[traffic_sign_i_frame0] = traffic_sign_indexs_frame1[nearby_traffic_signs[0]]
                else:
                    
                    nearby_traffic_signs = np.where(distance_to_frame1_traffic_signs_xy<1.0)[0] ##nearby_traffic_signs under traffic_sign_indexs_frame1
                    if Debug:
                        print('nearby sign B: ',nearby_traffic_signs)
                    if len(nearby_traffic_signs) == 1:
                        matchedseg_inframe0[traffic_sign_i_frame0] = traffic_sign_i_frame0
                        matchedseg_inframe1[traffic_sign_i_frame0] = traffic_sign_indexs_frame1[nearby_traffic_signs[0]]   
                        
    ################# matching traffic light  #########################
    traffic_light_indexs_frame0 = np.where(seg_class_indx_frame0 == global_config.traffic_light_class)[0]#index under  full segmentations  
    traffic_light_indexs_frame1 = np.where(seg_class_indx_frame1 == global_config.traffic_light_class)[0]#index under  full segmentations  
    if len(traffic_light_indexs_frame0)>0:
        
        for traffic_light_i_frame0 in traffic_light_indexs_frame0:
            if (traffic_light_i_frame0 is None) or (traffic_light_indexs_frame1 is None):
                break
            if (valid_location_utm_frame0 is None) or (valid_location_utm_frame1 is None):
                break
            distance_to_frame1_traffic_lights = LA.norm(valid_location_utm_frame0[traffic_light_i_frame0,0:2] - valid_location_utm_frame1[traffic_light_indexs_frame1,0:2], axis=1)
            nearby_traffic_lights = np.where(distance_to_frame1_traffic_lights<4)[0]# nearby_traffic_lights is under traffic_light_indexs_frame1
            
            if len(nearby_traffic_lights)>0:
                
                if len(nearby_traffic_lights)==1:
                    if distance_to_frame1_traffic_lights[nearby_traffic_lights[0]] < 1.5 \
                        and abs(valid_location_utm_frame0[traffic_light_i_frame0, 2] -\
                            valid_location_utm_frame1[traffic_light_indexs_frame1[nearby_traffic_lights[0]],2]) <2:  
                        
                        matchedseg_inframe0[traffic_light_i_frame0] = traffic_light_i_frame0
                        matchedseg_inframe1[traffic_light_i_frame0] = traffic_light_indexs_frame1[nearby_traffic_lights[0]]
                else:
                    
                    nearby_traffic_lights = np.where(distance_to_frame1_traffic_lights<0.8)[0]
                    if len(nearby_traffic_lights) == 1:
                        if distance_to_frame1_traffic_lights[nearby_traffic_lights[0]] < 0.8 \
                        and abs(valid_location_utm_frame0[traffic_light_i_frame0, 2] -\
                            valid_location_utm_frame1[traffic_light_indexs_frame1[nearby_traffic_lights[0]],2]) <2:  
                        
                            matchedseg_inframe0[traffic_light_i_frame0] = traffic_light_i_frame0
                            matchedseg_inframe1[traffic_light_i_frame0] = traffic_light_indexs_frame1[nearby_traffic_lights[0]] 
                        
    ################# matching pole #########################
    vegetation_indexs_frame0 = np.where(seg_class_indx_frame0 == global_config.vegetation_class)[0]#index under  full segmentations  
    vegetation_indexs_frame1 = np.where(seg_class_indx_frame1 == global_config.vegetation_class)[0]#index under  full segmentations  
    if len(vegetation_indexs_frame0)>0:
        
        for vegetation_i_frame0 in vegetation_indexs_frame0:
            if (vegetation_i_frame0 is None) or (vegetation_indexs_frame1 is None):
                break
            if (valid_location_utm_frame0 is None) or (valid_location_utm_frame1 is None):
                break
            distance_to_frame1_vegetations = LA.norm(valid_location_utm_frame0[vegetation_i_frame0,0:2] - valid_location_utm_frame1[vegetation_indexs_frame1,0:2], axis=1)
            nearby_vegetations = np.where(distance_to_frame1_vegetations<2)[0]
            
            if len(nearby_vegetations)>0:#nearby_vegetations is under vegetation_indexs_frame1
                
                if len(nearby_vegetations)==1:
                        matchedseg_inframe0[vegetation_i_frame0] = vegetation_i_frame0
                        matchedseg_inframe1[vegetation_i_frame0] = vegetation_indexs_frame1[nearby_vegetations[0]]

    ################ matching window #########################
    window_indexs_frame0 = np.where(seg_class_indx_frame0 == global_config.windows_class)[0]#index under full segmentations  
    window_indexs_frame1 = np.where(seg_class_indx_frame1 == global_config.windows_class)[0]#index under full segmentations  
    
    if len(window_indexs_frame0)>0:
        
        for window_i_frame0 in window_indexs_frame0:
            if (window_i_frame0 is None) or (window_indexs_frame1 is None):
                break
            if (valid_location_utm_frame0 is None) or (valid_location_utm_frame1 is None):
                break
            distance_to_frame1_windows_xy = LA.norm(valid_location_utm_frame0[window_i_frame0,0:2] - valid_location_utm_frame1[window_indexs_frame1,0:2], axis=1)
            distance_to_frame1_windows_z = np.abs(valid_location_utm_frame0[window_i_frame0,2] - valid_location_utm_frame1[window_indexs_frame1,2])
            
            distance_to_frame1_windows_xyz = np.logical_and(distance_to_frame1_windows_xy < 0.8,
                                                                 distance_to_frame1_windows_z < 1.5)
            nearby_windows = np.where(distance_to_frame1_windows_xyz)[0]#index under window_indexs_frame1
            if Debug:
                print('nearby wins' ,nearby_windows)
            
            if len(nearby_windows)==1:
                matchedseg_inframe0[window_i_frame0] = window_i_frame0
                matchedseg_inframe1[window_i_frame0] = window_indexs_frame1[nearby_windows[0]]
                    
        already_matched_win_frame0 = np.where(matchedseg_inframe0[window_indexs_frame0]>=0)[0] ###index under window_indexs_frame0
        unmatched_win_frame0 = np.where(matchedseg_inframe0[window_indexs_frame0]<0)[0] ###index under window_indexs_frame0
        
        if len(already_matched_win_frame0)>0 and len(unmatched_win_frame0)>0:
            a = window_indexs_frame1
            b = matchedseg_inframe1[window_indexs_frame0[already_matched_win_frame0]]
            unmatched_win_frame1 = np.setdiff1d(a, b) ###index under full segmentations
            
            
            for unmatched_window_i_frame0 in window_indexs_frame0[unmatched_win_frame0]:
                #unmatched_window_i_frame0 is index under  full segmentations  
                if Debug:
                    print('unmatched window i frame0', unmatched_window_i_frame0)
                
                distance_win = LA.norm(valid_location_utm_frame0[unmatched_window_i_frame0,...] - valid_location_utm_frame1[unmatched_win_frame1,...], axis=1)
                if len(distance_win)>0:
                    nearestwin_index = np.argmin(distance_win) 
                    nearest_dis = distance_win[nearestwin_index]
                    if nearest_dis < 2.2:
                        matchedseg_inframe0[unmatched_window_i_frame0] = unmatched_window_i_frame0
                        matchedseg_inframe1[unmatched_window_i_frame0] = unmatched_win_frame1[nearestwin_index]

    ############## delete repeated matched segs ###################################
    matched_unique = [i for i in set(matchedseg_inframe1) if i != -100 and i != -10]
    matchedseg_inframe1 = np.array(matchedseg_inframe1)
    matchedseg_inframe0 = np.array(matchedseg_inframe0)
    match_repeat = [j for j in matched_unique if len(np.where(matchedseg_inframe1 == j)[0]) > 1]
    for j in match_repeat:
        ttt = np.where(matchedseg_inframe1 == j)[0]
        matchedseg_inframe1[ttt] = -10  
        matchedseg_inframe0[ttt] = -10

    if global_config.ImagePatchBackgroud:
        image_path = os.path.join(global_config.image_dir, str(image_frame_time0) + '.png')
        image_ori0 = load_image(image_path, global_config.imagemodel)
        image_path = os.path.join(global_config.image_dir, str(image_frame_time1) + '.png')
        image_ori1 = load_image(image_path, global_config.imagemodel)

        image_ori0 = seq.augment_images([image_ori0.astype(np.uint8)])
        image_ori0 = image_ori0[0]
        image_ori1 = seq.augment_images([image_ori1.astype(np.uint8)])
        image_ori1 = image_ori1[0]
        
        image_curr = np.array(image_ori0)
        image_next = np.array(image_ori1)
        num_rows = image_curr.shape[0]
        num_cols = image_curr.shape[1]
        num_rows_next = image_next.shape[0]
        num_cols_next = image_next.shape[1]
        
        images_frame0_new = []
        images_frame1_new = []
        seg_pixel_coors_0 = []
        seg_pixel_coors_1 = []
        for index_i in range(len(images_frame0)):
            up, down, left, right = seg_image_coord_frame0[index_i,...]
            seg_pixel_coors = np.array([[up, left], 
                                        [up, right],
                                        [down, left],
                                        [down, right]]).astype(np.float32)
            if (left-Value_padding>=0 and right+Value_padding<=num_cols) and (up-Value_padding>=0 and down+Value_padding<=num_rows):
                ori_seg = patch_w_background(up-Value_padding, down+Value_padding, left-Value_padding, right+Value_padding, image_ori0)
            else:
                ori_seg = patch_w_background(up, down, left, right, image_ori0)
            images_frame0_new.append(patch_transform(ori_seg))
            seg_pixel_coors_0.append(seg_pixel_coors)
        for index_i in range(len(images_frame1)):
            up, down, left, right = seg_image_coord_frame1[index_i,...]
            seg_pixel_coors = np.array([[up, left], 
                                        [up, right],
                                        [down, left],
                                        [down, right]]).astype(np.float32)
            if (left-Value_padding>=0 and right+Value_padding<=num_cols_next) and (up-Value_padding>=0 and down+Value_padding<=num_rows_next):
                ori_seg = patch_w_background(up-Value_padding, down+Value_padding, left-Value_padding, right+Value_padding, image_ori1)
            else:
                ori_seg = patch_w_background(up, down, left, right, image_ori1)
            
            images_frame1_new.append(patch_transform(ori_seg))
            seg_pixel_coors_1.append(seg_pixel_coors)
        images_frame0 = images_frame0_new
        images_frame1 = images_frame1_new
    if Debug:
        image_path = os.path.join(global_config.image_dir, str(image_frame_time0) + '.png')
        image_ori0 = load_image(image_path, global_config.imagemodel)
        image_path = os.path.join(global_config.image_dir, str(image_frame_time1) + '.png')
        image_ori1 = load_image(image_path, global_config.imagemodel)
        
        import matplotlib.colors as mcolors
        colorlist = list(mcolors.CSS4_COLORS)
        colorind = np.random.choice(len(colorlist), 50).tolist()
        colorlist = [colorlist[_] for _ in colorind]
        fig, ax = plt.subplots(figsize=(20, 10), dpi=180)
        ax.imshow(image_ori0)
        valid_location_utm_frame0 = np.around(valid_location_utm_frame0, 1)
        print(valid_location_utm_frame0)
        
        tempi = 0
        for imagei in range(len(images_frame0)):
            minr, maxr, minc, maxc = seg_image_coord_frame0[imagei, ...]
            if matchedseg_inframe0[imagei] < 0:
                color = 'red'
            else:
                color = colorlist[matchedseg_inframe0[imagei]]
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=color, linewidth=5)
            ax.add_patch(rect)
            ax.annotate(str(seg_class_indx_frame0[imagei]), (minc, maxr), color='yellow', fontsize=10, ha='center',
                        va='center')
            ax.annotate(str(valid_location_utm_frame0[imagei]), (minc, maxr - np.random.randint(maxr - minr)),
                        color='pink', weight='bold', fontsize=10, ha='center', va='center')
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        
        fig, ax1 = plt.subplots(figsize=(20, 10), dpi=180)
        ax1.imshow(image_ori1)
        valid_location_utm_frame1 = np.around(valid_location_utm_frame1, 1)
        print(valid_location_utm_frame1)

        for imagei in range(len(images_frame1)):
            minr, maxr, minc, maxc = seg_image_coord_frame1[imagei, ...]
            corlori = np.where(matchedseg_inframe1==imagei)[0]
            if len(corlori)==0:
                color = 'red' 
            else:
                print(imagei, corlori[0])
                color = colorlist[corlori[0]]
                
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=color, linewidth=5)
            
            ax1.add_patch(rect)

            ax1.annotate(str(seg_class_indx_frame1[imagei]), (minc, maxr), color='yellow', fontsize=10, ha='center',
                        va='center')
            ax1.annotate(str(valid_location_utm_frame1[imagei]), (minc, maxr - np.random.randint(maxr - minr)),
                        color='pink', weight='bold', fontsize=10, ha='center', va='center')

        ax1.set_axis_off()
        plt.tight_layout()
        plt.show()

    image_size = 256
    images_frame0 = torch.cat(images_frame0, dim=0).view(-1, 3, image_size, image_size)
    images_frame1 = torch.cat(images_frame1, dim=0).view(-1, 3, image_size, image_size)
    valid_location_utm_frame0 = torch.from_numpy(np.array(valid_location_utm_frame0))
    valid_location_utm_frame1 = torch.from_numpy(np.array(valid_location_utm_frame1))
    matchedseg_inframe0 = torch.IntTensor(matchedseg_inframe0)
    matchedseg_inframe1 = torch.IntTensor(matchedseg_inframe1)
    adj_full = torch.from_numpy(np.asarray(normalize_adj(np.ones((len(matchedseg_inframe1),len(matchedseg_inframe1))))).astype(np.float32))
    x = []
    car_location = []
    car_direction = []
    seg_feature = []
    seg_class_indx = []
    seg_lidaradj = []
    seg_dis2vehicle = []
    valid_location_frame = []
    
    x.append((images_frame1, car_location, car_direction, seg_feature, seg_class_indx, seg_lidaradj, seg_dis2vehicle, seg_pixel_coors_1, valid_location_frame, adj_full, matchedseg_inframe1)) 
    x.append((images_frame0, car_location, car_direction, seg_feature, seg_class_indx, seg_lidaradj, seg_dis2vehicle, seg_pixel_coors_0, valid_location_frame, adj_full, matchedseg_inframe1))
    return x

class LandmarksDataset_train(Dataset):
    def __init__(self, savedpath='', shift=False):

        self.shift = shift
        self.savedpath = savedpath
        self.valid_idx = np.load('*.npy',allow_pickle=True) 
    def __len__(self):
        return len(self.valid_idx)
    def __getitem__(self, idx):
        iidx = idx
        x = []
        while 1:
            image_frame_time0, image_frame_time1 = int(self.valid_idx[iidx, 0]), int(self.valid_idx[iidx, 1])
            x = get_graph_traning_testing(image_frame_time0, image_frame_time1)
            if x is not None:
                break
            else:
                iidx = np.random.randint(len(self.valid_idx))

        return x
		
class LandmarksDataset_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, savedpath='', shift=False):

        self.shift = shift
        self.savedpath = savedpath
        self.valid_idx = np.load('*.npy',allow_pickle=True) 
    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        iidx = idx
        x = []
        while 1:
            image_frame_time0, image_frame_time1 = int(self.valid_idx[iidx, 0]), int(self.valid_idx[iidx, 1])
            x = get_graph_traning_testing(image_frame_time0, image_frame_time1)
            if x is not None:
                break
            else:
                iidx = np.random.randint(len(self.valid_idx))

        return x
