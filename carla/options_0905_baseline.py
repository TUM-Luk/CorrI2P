import numpy as np
import math
import torch


class Options:
    def __init__(self):
        # data config
        self.dataroot = './dataset_large_int_train'
        self.train_subdir = 'mapping'
        self.val_subdir = 'query'
        self.test_subdir = 'query'
        
        self.train_txt = "dataset_large_int_train/train_list_deepi2p/train_75scene.txt"
        self.val_txt = "dataset_large_int_train/train_list_deepi2p/val_75scene_t3_int4.txt"
        self.test_txt = "dataset_large_int_train/train_list_deepi2p/val_75scene_t10_int1.txt"
        self.pin_memory = True
        
        self.vis_debug = False
        self.is_debug = False
        self.is_fine_resolution = True
        self.is_remove_ground = False
        self.accumulation_frame_num = 3
        self.accumulation_frame_skip = 6

        self.delta_ij_max = 40
        self.translation_max = 10.0

        self.crop_original_top_rows = 50
        self.img_scale = 1/3.75
        self.img_H = 288  
        self.img_W = 512  
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32
        self.num_kpt = 512
        self.is_front = False

        self.input_pt_num = 20480
        self.pc_min_range = -1.0
        self.pc_max_range = 40.0 # default corri2p for kitti: 80
        self.node_a_num = 128
        self.node_b_num = 128
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = 10
        self.P_ty_amplitude = 10
        self.P_tz_amplitude = 5
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 0.0 * math.pi / 12.0
        self.P_Rz_amplitude = 2.0 * math.pi
        self.dataloader_threads = 4

        # learning
        self.pos_margin = 0.2
        self.neg_margin = 1.8
        self.dist_thres = 1
        self.img_thres = 0.9
        self.pc_thres = 0.9
        
        self.batch_size = 8
        self.gpu_ids = [0]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.001 # 0.001
        self.min_lr = 0.00001
        self.lr_decay_step = 10
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4
        if self.is_fine_resolution:
            self.coarse_loss_alpha = 50
        else:
            self.coarse_loss_alpha = 1
            
        # args
        self.epoch = 250
        self.train_batch_size = 8
        self.val_batch_size = 1
        self.save_path = './runs_carla'
        self.save_name = "all_scene_baseline"
        
            




