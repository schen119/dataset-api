"""
This file is refered from https://github.com/stevenwudi/6DVNET/blob/master/tools/ApolloScape_car_instance/demo_mesh.py
Thanks for the author Stevenwudi
"""
import os
from tqdm import tqdm
from collections import namedtuple
from render_car_mesh import CarPoseVisualizer

Setting = namedtuple('Setting', ['image_name', 'data_dir'])

set_name = 'train-list'   #['train', 'val']
# You need to specify the dataset dir
#dataset_dir = '/media/SSD_1TB/ApolloScape/ECCV2018_apollo/train/'
data_dir = '/media/elvis/backup/dataset/apolloscape/3d_car_instance/train'
split_file = 'train-list' #[train-list, validation-list]


img_list = [line.rstrip('\n')[:-4] for line in open(os.path.join(data_dir, 'split', split_file + '.txt'))]
save_dir = os.path.join(data_dir, 'Mesh_overlay')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

setting = Setting(None, data_dir)
visualizer = CarPoseVisualizer(setting)
visualizer.load_car_models()

img_list = ['171206_034636094_Camera_5']
#img_list = ['180114_024339575_Camera_5']
car_pose_dir = os.path.join(data_dir, 'car_poses')

for img in tqdm(img_list):
    print("process img = {}".format(img))
    car_pose_file = os.path.join(car_pose_dir, '%s.json' % img)
    setting = Setting(img, data_dir)
    visualizer.set_dataset(setting)
    merged_image = visualizer.showAnn_image(setting.image_name, car_pose_file, set_name, save_dir)


