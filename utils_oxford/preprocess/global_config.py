import numpy as np
import numpy.matlib
import torch
import sys
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=10,sci_mode=False)
from raw_preprocess.python.camera_model_new import CameraModel
import pandas as pd
import os

homefolder = ''
camera_valid_times = np.load("*.npy")
lidar_valid_times = np.load("*.npy")
training_times = np.load("*.npy")
testing_times = np.load("*.npy")
image_dir = ''
seg_dir = ''
win_seg_dir = ''
models_dir = homefolder + 'preprocess/models'

imagemodel = CameraModel(models_dir, image_dir)
laser_dir = ""
poses_file = 'ins_gt_update_0.csv'

extrinsics_dir = homefolder + 'preprocess/extrinsics/'
poses_file_pd = None
image_extrinsics_path = os.path.join(extrinsics_dir, imagemodel.camera + '.txt')
validmax_map_rotation = 0.3
validmax_map_length = 15
max_num_image_frame = 5
max_num_lidar_frame = 30

minimum_pixel_up = 150  
minimum_pixel_low = 10
maximum_pixel_up = 5000  
maximum_pixel_low = 2500
minimum_pixel = 150  
maximum_pixel = 10000000  
Thre_frame_i_object = 40
Thre_pixel_coor_len = 500
Threshold_valid_location_utm_traffic_light = 0.8
Threshold_valid_location_utm_traffic_pole = 0.8
Threshold_valid_location_utm_window_width = 0.8
Threshold_valid_location_utm_window_height = 0.8

ImagePatchBackgroud = True
num_valid_lidar_points = 2  

origin = np.array([675017.13719174, 5406632.4742808 ])
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
num_pixel = 0
image_size = 256


class_num = 1000
feature_dim = 4
class_feature = np.random.uniform(-1,1,(class_num, feature_dim))
x_min = 5734750
y_min = 619570
z_min = -114

win_edge_width = 10

unlabeled_class            =   255       
ego_vehicle_class          =   255       
rectification_border_class =   255       
out_of_roi_class           =   255       
static_class               =   255       
dynamic_class              =   255       
ground_class               =   255       
road_class                 =     0       
sidewalk_class             =     1       
parking_class              =   255       
rail_track_class           =   255       
building_class             =     2       
wall_class                 =     3       
fence_class                =     4       
guard_rail_class           =   255       
bridge_class               =   255       
tunnel_class               =   255       
pole_class                 =     5       
polegroup_class            =   255       
traffic_light_class        =     6       
traffic_sign_class         =     7       
vegetation_class           =     8       
terrain_class              =     9       
sky_class                  =    10       
person_class               =    11       
rider_class                =    12       
car_class                  =    13       
truck_class                =    14       
bus_class                  =    15       
caravan_class              =   255       
trailer_class              =   255       
train_class                =    16       
motorcycle_class           =    17       
bicycle_class              =    18       
license_plate_class        =    -1      
windows_class = 1000



