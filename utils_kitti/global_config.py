import numpy as np
import numpy.matlib
import torch
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=10,sci_mode=False)
origin = np.array([ 675017.13719174, 5406632.4742808 ])
image_mean =np.asarray([0.43430279, 0.41444465, 0.38931389])
image_std = np.asarray([0.08568499, 0.08338873, 0.08511615])
num_pixel = 0
class_num = 1000
temp = np.arange(class_num)
feature_dim = 4
class_feature = np.random.uniform(-1,1,(class_num, feature_dim))
num_rows = 768
num_cols = 1024
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
windows_class = 10000
             
len_seq = [
4541,
1101,
4661,
801,
271,
2761,
1101,
1101,
4071,
1591,
1201]

