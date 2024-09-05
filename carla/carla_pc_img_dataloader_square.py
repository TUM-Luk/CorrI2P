import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
from kapture.io.csv import kapture_from_dir
import tqdm
import quaternion
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader,
    Sampler
)
from carla.depth_convert import dpt_3d_convert

class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx

def make_carla_dataloader(mode, opt):
    data_root = opt.dataroot  # 'dataset_large_int_train'
    
    train_subdir = opt.train_subdir  # 'mapping'
    val_subdir = opt.val_subdir  # 'mapping'
    test_subdir = opt.test_subdir  # 'query'
    
    train_txt = opt.train_txt  # "dataset_large_int_train/train_list/train_t1_int1_v50_s25_io03_vo025.txt"
    val_txt = opt.val_txt 
    test_txt = opt.test_txt
    
    
    if mode == 'train':
        data_txt = train_txt
    elif mode == 'val':
        data_txt = val_txt
    elif mode == 'test':
        data_txt = test_txt
        
    with open(data_txt, 'r') as f:
        voxel_list = f.readlines()
        voxel_list = [voxel_name.rstrip() for voxel_name in voxel_list]
        
    kapture_datas={}
    sensor_datas={}
    input_path_datas={}
    train_list_kapture_map={}
    for train_path in voxel_list:
        # scene=os.path.dirname(os.path.dirname(train_path))
        scene=train_path.split('/')[0]
        if scene not in kapture_datas:
            if mode=='test':
                input_path=os.path.join(data_root,scene, test_subdir)
            elif mode=='train':
                input_path=os.path.join(data_root,scene, train_subdir)
            else:
                input_path=os.path.join(data_root, scene, val_subdir)
            kapture_data=kapture_from_dir(input_path)
            sensor_dict={}
            for timestep in kapture_data.records_camera:
                _sensor_dict=kapture_data.records_camera[timestep]
                for k, v in _sensor_dict.items():
                    sensor_dict[v]=(timestep, k)
            kapture_datas[scene]=kapture_data
            sensor_datas[scene]=sensor_dict
            input_path_datas[scene]=input_path
        train_list_kapture_map[train_path]=(kapture_datas[scene], sensor_datas[scene], input_path_datas[scene])
        
    datasets = []
    
    for train_path in tqdm.tqdm(voxel_list):
        kapture_data, sensor_data, input_path=train_list_kapture_map[train_path]
        one_dataset = carla_pc_img_dataset(root_path=data_root, train_path=train_path, mode=mode, opt=opt,
                                kapture_data=kapture_data, sensor_data=sensor_data, input_path=input_path)
        
        one_dataset[10]
        datasets.append(one_dataset)
        
    
    final_dataset = ConcatDataset(datasets)
    
    if mode=='train':
        dataloader = DataLoader(final_dataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True,
                                num_workers=opt.dataloader_threads, pin_memory=opt.pin_memory
                                )
    elif mode=='val' or mode=='test':
        dataloader = DataLoader(final_dataset, batch_size=opt.val_batch_size, shuffle=False,
                                num_workers=opt.dataloader_threads, pin_memory=opt.pin_memory
                                )
    else:
        raise ValueError
    
    return final_dataset, dataloader

class carla_pc_img_dataset(data.Dataset):
    def __init__(self, root_path, train_path, mode, opt,
                 kapture_data, sensor_data, input_path):
        super(carla_pc_img_dataset, self).__init__()
        
        self.root_path = root_path  # "dataset_large_int_train"
        self.train_path = train_path  # 't1_int1'
        self.mode = mode
        self.opt = opt
        
        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)
        
        self.sensor_dict = sensor_data
        self.kaptures = kapture_data
        self.input_path = input_path
   
        self.num_pc = opt.input_pt_num
        self.img_H = opt.img_H
        self.img_W = opt.img_W

        self.P_tx_amplitude = opt.P_tx_amplitude
        self.P_ty_amplitude = opt.P_ty_amplitude
        self.P_tz_amplitude = opt.P_tz_amplitude
        self.P_Rx_amplitude = opt.P_Rx_amplitude
        self.P_Ry_amplitude = opt.P_Ry_amplitude
        self.P_Rz_amplitude = opt.P_Rz_amplitude
        self.num_kpt=opt.num_kpt
        self.farthest_sampler = FarthestSampler(dim=3)

        self.node_a_num=opt.node_a_num 
        self.node_b_num=opt.node_b_num 
        self.is_front=opt.is_front
        
        self.dataset = self.make_carla_dataset(root_path, train_path, mode)
        self.voxel_points = self.make_voxel_pcd()
        
        self.projector = dpt_3d_convert()
        
        print(f'load data complete. {len(self.dataset)} image-voxel pair')

    def make_carla_dataset(self, root_path, train_path, mode):
        dataset = []

        if mode == "train":
            dataset = list(self.sensor_dict.keys())
        elif mode == "val" or mode == "test":
            dataset = list(self.sensor_dict.keys())
        else:
            raise ValueError
        
        return dataset
    
    def make_voxel_pcd(self):
        scene_name = self.train_path.split('/')[0]
        point_cloud_file = os.path.join(self.input_path, f'pcd_{scene_name}_train_down.ply')
        print(f"load pcd file from {point_cloud_file}")
        pcd = open3d.io.read_point_cloud(point_cloud_file)
        pcd_points = np.array(pcd.points)
        pcd_points = pcd_points.astype(np.float32)

        voxel_points = pcd_points.T  # 整个场景（路口）的点云
        
        return voxel_points


    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        if intensity_max == 0:
            fake_colors[:, 0:1] = np.transpose(intensity)
        else:
            fake_colors[:, 0:1] = np.transpose(intensity)/intensity_max

        pcd.colors=open3d.utility.Vector3dVector(fake_colors)
        pcd.normals=open3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals))

        return pointcloud, intensity, sn

    def downsample_np(self, pc_np, intensity_np, sn_np, num_pc):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        # print('t',t)
        # print('angles',angles)

        return P_random

    def jitter_point_cloud(self, pc_np, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
            CxN array, original point clouds
            Return:
            CxN array, jittered point clouds
        """
        C, N = pc_np.shape
        assert(clip > 0)
        jittered_pc = np.clip(sigma * np.random.randn(C, N), -1*clip, clip).astype(pc_np.dtype)
        jittered_pc += pc_np
        return jittered_pc

    # def augment_pc(self, pc_np, intensity_np):
    #     """

    #     :param pc_np: 3xN, np.ndarray
    #     :param intensity_np: 3xN, np.ndarray
    #     :param sn_np: 1xN, np.ndarray
    #     :return:
    #     """
    #     # add Gaussian noise
    #     pc_np = self.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
    #     intensity_np = self.jitter_point_cloud(intensity_np, sigma=0.01, clip=0.05)
    #     return pc_np, intensity_np
    
    def __len__(self):
        return len(self.dataset)
    
    def load_pose(self, timestep, sensor_id):
        if self.kaptures.trajectories is not None and (timestep, sensor_id) in self.kaptures.trajectories:
            pose_world_to_cam = self.kaptures.trajectories[(timestep, sensor_id)]
            pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
            pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
            pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
            pose_world_to_cam_matrix[3, 3] = 1.0
            T = torch.tensor(pose_world_to_cam_matrix).float()
            gt_pose=T.inverse() # gt_pose为从cam_to_world
        else:
            gt_pose=T=torch.eye(4)
        return gt_pose, pose_world_to_cam
    
    def __getitem__(self, index):
        image_id = self.dataset[index]
        timestep, sensor_id=self.sensor_dict[image_id]
        
        # camera intrinsics
        camera_params=np.array(self.kaptures.sensors[sensor_id].camera_params[2:])
        K = np.array([[camera_params[0],0,camera_params[1]],
                    [0,camera_params[0],camera_params[2]],
                    [0,0,1]])
        
        # T from point cloud to camera
        gt_pose, gt_pose_world_to_cam_q=self.load_pose(timestep, sensor_id) # camera to world
        gt_pose_world_to_cam_q = np.concatenate((gt_pose_world_to_cam_q.t_raw, gt_pose_world_to_cam_q.r_raw))
        T_c2w = gt_pose.numpy() # camera to world
        T_w2c = np.linalg.inv(T_c2w)
        
        # T from world to voxel coordinate 将坐标系移到camera附近，高度不动
        T_w2v = np.eye(4).astype(np.float32)
        T_w2v[:2,3] = -T_c2w[:2,3]
        T_w2v_inv = np.linalg.inv(T_w2v).copy()
        
        # ------------- load image, original size is 1080x1920 -------------
        img = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id))
        depth_map_path = os.path.join(self.input_path, 'sensors/depth_data', image_id.replace("image", "depth"))
        depth_map = Image.open(depth_map_path) # RGBA
        depth_map = np.array(depth_map)
        
        R = depth_map[:,:,0].astype(np.float32)
        G = depth_map[:,:,1].astype(np.float32)
        B = depth_map[:,:,2].astype(np.float32)
        normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        depth_map = 1000 * normalized

        # origin image
        if self.opt.vis_debug:
            cv2.imwrite(f'z_dataset/img_ori_{index}.png', img)
        
        # scale to 360x640
        new_size = (int(round(img.shape[1] * self.opt.img_scale)), int(round((img.shape[0] * self.opt.img_scale))))
        img = cv2.resize(img,
                         new_size,
                         interpolation=cv2.INTER_LINEAR)
        depth_map_image = Image.fromarray(depth_map)
        resized_depth_map_nearest = depth_map_image.resize(new_size, Image.NEAREST)
        depth_map = np.array(resized_depth_map_nearest)
        K = self.camera_matrix_scaling(K, self.opt.img_scale)
        
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        depth_map = depth_map[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)
        
        # resize and cropped image
        if self.opt.vis_debug:
            cv2.imwrite(f'z_dataset/img_resize_crop_{index}.png', img)
            
        # ------------- load point cloud ----------------
        npy_data = self.voxel_points.copy() # important! keep self.voxel points unchanged
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = np.zeros((1, pc_np.shape[1]), dtype=np.float32)  # 1xN
        surface_normal_np = np.zeros((3, pc_np.shape[1]), dtype=np.float32)  # 3xN

        # origin pcd
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_{index}.ply', debug_point_cloud)
        
        # transform frame to voxel center
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(T_w2v, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_{index}.ply', debug_point_cloud)

        # limit max_z, the pc is in CAMERA coordinate
        pc_np_x_square = np.square(pc_np[0, :])
        pc_np_y_square = np.square(pc_np[1, :])
        pc_np_range_square = pc_np_x_square + pc_np_y_square
        pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
        pc_np = pc_np[:, pc_mask_range]
        intensity_np = intensity_np[:, pc_mask_range]
        surface_normal_np = surface_normal_np[:, pc_mask_range]
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_limit_{index}.ply', debug_point_cloud)
        
        # point cloud too huge, voxel grid downsample first
        if pc_np.shape[1] > 4 * self.opt.input_pt_num:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np, surface_normal_np = self.downsample_with_intensity_sn(pc_np, intensity_np, surface_normal_np, voxel_grid_downsample_size=0.4)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
            surface_normal_np = surface_normal_np.astype(np.float32)
            
        # random sampling
        pc_np, intensity_np, surface_normal_np = self.downsample_np(pc_np, intensity_np, surface_normal_np, self.opt.input_pt_num)
        
        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform()
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            # pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            if random.random() > 0.5:
                img = self.augment_img(img)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        elif 'val' == self.mode or 'test' == self.mode:
            Pr = np.identity(4, dtype=np.float32)
            Pr_inv = np.identity(4, dtype=np.float32)

        
        P = T_w2c @ T_w2v_inv @ Pr_inv # 对于输入点云的新GT Pose
        
        # then aug to get final input pcd
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(Pr, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_T_w2v_aug_{index}.ply', debug_point_cloud)
        
        # input pcd in cam coordinate frame
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(P, pc_homo_np)  # 4xN
        pc_np_in_cam = Pr_pc_homo_np[0:3, :]  # 3xN
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np_in_cam.T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_in_cam_{index}.ply', debug_point_cloud)

        
        #1/4 scale
        K_4=self.camera_matrix_scaling(K,0.25)
        
        pc_ = np.dot(K_4, pc_np_in_cam)
        pc_mask = np.zeros((1, np.shape(pc_np_in_cam)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.floor(pc_[0:2, :]).astype(np.int64)
        
        depth_map_image = Image.fromarray(depth_map)
        resized_depth_map_nearest = depth_map_image.resize((int(self.img_W*0.25), int(self.img_H*0.25)), Image.NEAREST)
        depth_map = np.array(resized_depth_map_nearest)
        min_depth = depth_map[np.clip(xy[1,:], 0, int(self.img_H*0.25 - 1)), np.clip(xy[0,:], 0, int(self.img_W*0.25 - 1))]
        depth_thres = 3
        
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) \
                        & (xy[1, :] >= 0) & (xy[1, :] <= (self.img_H*0.25 - 1)) \
                        & (pc_[2, :] > 0)
        is_in_picture_no_occ = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) \
                                & (xy[1, :] >= 0) & (xy[1, :] <= (self.img_H*0.25 - 1)) \
                                & (pc_[2, :] > 0) & (pc_[2, :] < min_depth + depth_thres)
                
        # pc_mask[:, is_in_picture] = 1.
        pc_mask[:, is_in_picture_no_occ] = 1.
        
        # in picture points
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(pc_np[:,is_in_picture].T)
            open3d.io.write_point_cloud(f'z_dataset/input_pcd_in_picture_{index}.ply', debug_point_cloud)
            
            
        pc_kpt_idx=np.where(is_in_picture_no_occ==1)[0]
        if len(pc_kpt_idx) >= self.num_kpt:
            indices=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt]
        else:
            print(f"in image pc no occ only {len(pc_kpt_idx)}")
            fix_idx = np.asarray(range(len(pc_kpt_idx)))
            while len(pc_kpt_idx) + fix_idx.shape[0] < self.num_kpt:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(len(pc_kpt_idx)))), axis=0)
            random_idx = np.random.choice(len(pc_kpt_idx), self.num_kpt - fix_idx.shape[0], replace=False)
            indices = np.concatenate((fix_idx, random_idx), axis=0)
        pc_kpt_idx=pc_kpt_idx[indices]

        pc_outline_idx=np.where(is_in_picture==0)[0]
        if len(pc_outline_idx) >= self.num_kpt:
            indices=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt]
        else:
            print(f"out frustum pc only {len(pc_outline_idx)}")
            fix_idx = np.asarray(range(len(pc_outline_idx)))
            while len(pc_outline_idx) + fix_idx.shape[0] < self.num_kpt:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(len(pc_outline_idx)))), axis=0)
            random_idx = np.random.choice(len(pc_outline_idx), self.num_kpt - fix_idx.shape[0], replace=False)
            indices = np.concatenate((fix_idx, random_idx), axis=0)
        pc_outline_idx=pc_outline_idx[indices]

        # xy2 = xy[:, is_in_picture]
        xy2 = xy[:, is_in_picture_no_occ]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(self.img_H*0.25), int(self.img_W*0.25))).toarray()
        img_mask = np.array(img_mask)
        img_mask[img_mask > 0] = 1. # img_mask表示哪些地方有被点投影到

        # use depth projection to get img_mask for img_outline_index
        resize_w = img_mask.shape[1]
        resize_h = img_mask.shape[0]
        x = np.arange(resize_w)
        y = np.arange(resize_h)
        xv, yv = np.meshgrid(x, y)

        keypoints = np.vstack([xv.ravel(), yv.ravel()]).T
        keypoints = keypoints.astype(np.int16)
        depths = depth_map.reshape(-1)
        depth_mask = depths < 100
        # project into 3D points
        dense_point = self.projector.proj_2to3(keypoints, depths, K_4, T_w2v@T_c2w, depth_unit=1)
        if self.opt.vis_debug:
            debug_point_cloud = open3d.geometry.PointCloud()
            debug_point_cloud.points = open3d.utility.Vector3dVector(dense_point[depth_mask])
            open3d.io.write_point_cloud(f'z_dataset/dense_point_{index}.ply', debug_point_cloud)
        dense_point = dense_point.reshape(resize_h, resize_w, 3).astype(np.float32)
        img_dist = np.linalg.norm(dense_point, axis=2)
        img_mask = img_dist <= self.opt.pc_max_range # 所有投影到3D点后，在输入点云范围内的点
        
        # in voxel mask
        if self.opt.vis_debug:
            debug_img_mask = (img_mask*255).astype(np.uint8)
            cv2.imwrite(f'z_dataset/img_mask_{index}.png', debug_img_mask)
            
        img_kpt_index=xy[1,pc_kpt_idx]*self.img_W*0.25 +xy[0,pc_kpt_idx] # 铺平后，512个点对应的keypoint的下标


        img_outline_index=np.where(img_mask.squeeze().reshape(-1)==0)[0]
        if len(img_outline_index) < self.num_kpt:
            img_outline_index=np.argsort(img_dist.reshape(-1))[-self.num_kpt:]
        else:
            indices=np.random.permutation(len(img_outline_index))[0:self.num_kpt]
            img_outline_index=img_outline_index[indices] # 选取512个没被点投影到的像素的铺平下标

        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice( pc_np.shape[1],
                                                                            self.node_a_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_a_num)

        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice( pc_np.shape[1],
                                                                            self.node_b_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_b_num)

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc_np.astype(np.float32)),
                'pc_np_in_cam': torch.from_numpy(pc_np_in_cam.astype(np.float32)),
                'intensity': torch.from_numpy(intensity_np.astype(np.float32)),
                'sn': torch.from_numpy(surface_normal_np.astype(np.float32)),
                'K': torch.from_numpy(K_4.astype(np.float32)),
                'P': torch.from_numpy(P.astype(np.float32)),

                'pc_mask': torch.from_numpy(pc_mask).float(),       #(1,20480)
                'img_mask': torch.from_numpy(img_mask).float(),     #(40,128)
                
                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),         #512
                'pc_outline_idx':torch.from_numpy(pc_outline_idx),  #512
                'img_kpt_idx':torch.from_numpy(img_kpt_index).long() ,      #512
                'img_outline_index':torch.from_numpy(img_outline_index).long(),
                'node_a':torch.from_numpy(node_a_np).float(),
                'node_b':torch.from_numpy(node_b_np).float()
                }
               


if __name__ == '__main__':
    from carla import options as options
    opt = options.Options()
    debug_dataset, debug_dataloader = make_carla_dataloader(mode="train", opt=opt)
    debug_dataset, debug_dataloader = make_carla_dataloader(mode="val", opt=opt)
    debug_dataset, debug_dataloader = make_carla_dataloader(mode="test", opt=opt)
    
    # dataset = carla_pc_img_dataset('/gpfs1/scratch/siyuren2/dataset/', 'val', 20480)
    # data = dataset[4000]
    
    '''img=data[0].numpy()              #full size
    pc=data[1].numpy()
    intensity=data[2].numpy()
    sn=data[3].numpy()
    K=data[4].numpy()
    P=data[5].numpy()
    pc_mask=data[6].numpy()      
    img_mask=data[7].numpy()    #1/4 size

    pc_kpt_idx=data[8].numpy()                #(B,512)
    pc_outline_idx=data[9].numpy()
    img_kpt_idx=data[10].numpy()
    img_outline_idx=data[11].numpy()

    np.save('./test_data/img.npy',img)
    np.save('./test_data/pc.npy',pc)
    np.save('./test_data/intensity.npy',intensity)
    np.save('./test_data/sn.npy',sn)
    np.save('./test_data/K.npy',K)
    np.save('./test_data/P.npy',P)
    np.save('./test_data/pc_mask.npy',pc_mask)
    np.save('./test_data/img_mask.npy',img_mask)
    '''



    '''for i,data in enumerate(dataset):
        print(i,data['pc'].size())'''

    # print(len(dataset))
    # print(data['pc'].size())
    # print(data['img'].size())
    # print(data['pc_mask'].size())
    # print(data['intensity'].size())
    # np.save('./test_data/pc.npy', data['pc'].numpy())
    # np.save('./test_data/P.npy', data['P'].numpy())
    # np.save('./test_data/img.npy', data['img'].numpy())
    # np.save('./test_data/img_mask.npy', data['img_mask'].numpy())
    # np.save('./test_data/pc_mask.npy', data['pc_mask'].numpy())
    # np.save('./test_data/K.npy', data['K'].numpy())
    

    """
    img = dict['img'].numpy()
    img_mask = dict['img_mask'].numpy()
    img = img.transpose(1, 2, 0)
    cv2.imwrite('img.png',np.uint8(img*255))
    cv2.imwrite('img_mask.png', np.uint8(img_mask * 255))
    cv2.imshow('img', cv2.resize(img,(512,160)))
    cv2.imshow('img_mask', cv2.resize(img_mask,(512,160)))

    color = []
    for i in range(np.shape(pc_data)[1]):
        if pc_mask[0, i] > 0:
            color.append([0, 1, 1])
        else:
            color.append([0, 0, 1])
    color = np.asarray(color, dtype=np.float64)
    print(color.shape)

    print(np.sum(pc_mask), np.sum(img_mask))
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc_data.T)
    pointcloud.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([pointcloud])
    # plt.imshow(dict['img'].permute(1,2,0).numpy())
    # plt.show()"""
