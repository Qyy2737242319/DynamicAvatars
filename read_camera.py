import json
import os
import numpy as np
from gaussiansplatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scipy.spatial.transform import Rotation as R

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers) # （3， 301）
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center # （3， 1）
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers) # 所有相机的均值中心点，最远距离
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

data = np.load('data/canonical_flame_param.npz')
a=[]
b=[4,5,6]
c=a+b
with open('./data/transforms_train.json', 'r') as f:
    json_data = json.load(f)

for i in range(len(json_data['camera_indices'])):
    camera_matrix = np.array(json_data['frames'][i]['transform_matrix'])
    camera_matrix = np.linalg.inv(camera_matrix)
    
    Rot = camera_matrix[:3, :3]
    Trans = camera_matrix[:3, 3]
    fl_x = json_data['frames'][i]['fl_x']
    fl_y = json_data['frames'][i]['fl_y']
    Fov_x = focal2fov(fl_x, 550)
    Fov_y = focal2fov(fl_y,     802)
    r = R.from_matrix(Rot)
    qvec = r.as_quat()
    w_qvec = np.array([-qvec[3],qvec[0],qvec[1],qvec[2]])
pass