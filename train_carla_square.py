import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import argparse
from network import CorrI2P
# import loss
from loss import desc_loss, det_loss2
import numpy as np
import datetime
import logging
import math
import cv2
from scipy.spatial.transform import Rotation

import numpy as np
import tqdm
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter


def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff


def test_acc(model,testdataloader,epoch,opt):
    
    t_diff_set=[]
    angles_diff_set=[]

    for step,data in tqdm.tqdm(enumerate(testdataloader), total=len(testdataloader)):
        # if step%1==0:
        with torch.no_grad():
            model.eval()
            img=data['img'].cuda()              #full size
            pc=data['pc'].cuda()
            origin_pc=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()      
            img_mask=data['img_mask'].cuda()    #1/4 size

            pc_kpt_idx=data['pc_kpt_idx'].cuda()                #(B,512)
            pc_outline_idx=data['pc_outline_idx'].cuda()
            img_kpt_idx=data['img_kpt_idx'].cuda()
            img_outline_idx=data['img_outline_index'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()

            img_features,pc_features,img_score,pc_score=model(pc,intensity,sn,img,node_a,node_b)     #64 channels feature
            
            img_score=img_score[0].data.cpu().numpy()
            pc_score=pc_score[0].data.cpu().numpy()
            img_feature=img_features[0].data.cpu().numpy()
            pc_feature=pc_features[0].data.cpu().numpy()
            pc=pc[0].data.cpu().numpy()
            origin_pc=origin_pc[0].data.cpu().numpy()
            P=P[0].data.cpu().numpy()
            K=K[0].data.cpu().numpy()
            
            img_x=np.linspace(0,np.shape(img_feature)[-1]-1,np.shape(img_feature)[-1]).reshape(1,-1).repeat(np.shape(img_feature)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
            img_y=np.linspace(0,np.shape(img_feature)[-2]-1,np.shape(img_feature)[-2]).reshape(-1,1).repeat(np.shape(img_feature)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])

            img_xy=np.concatenate((img_x,img_y),axis=0)

            img_xy_flatten=img_xy.reshape(2,-1)
            img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
            img_score_flatten=img_score.squeeze().reshape(-1)

            img_index=(img_score_flatten>opt.img_thres)
            if img_index.sum() < 50:
                topk_img_index=np.argsort(-img_score_flatten)[:opt.num_kpt]
                img_xy_flatten_sel=img_xy_flatten[:,topk_img_index]
                img_feature_flatten_sel=img_feature_flatten[:,topk_img_index]
                img_score_flatten_sel=img_score_flatten[topk_img_index]
            else:
                img_xy_flatten_sel=img_xy_flatten[:,img_index]
                img_feature_flatten_sel=img_feature_flatten[:,img_index]
                img_score_flatten_sel=img_score_flatten[img_index]

            pc_index=(pc_score.squeeze()>opt.pc_thres)
            if pc_index.sum() < 50:
                topk_pc_index=np.argsort(-pc_score.squeeze())[:opt.num_kpt]
                pc_sel=origin_pc[:,topk_pc_index]
                pc_feature_sel=pc_feature[:,topk_pc_index]
                pc_score_sel=pc_score.squeeze()[topk_pc_index]
            else:
                pc_sel=origin_pc[:,pc_index]
                pc_feature_sel=pc_feature[:,pc_index]
                pc_score_sel=pc_score.squeeze()[pc_index]

            # dist=1-np.sum(np.expand_dims(pc_feature_sel,axis=2)*np.expand_dims(img_feature_flatten_sel,axis=1),axis=0)
            dist = 1 - np.matmul(pc_feature_sel.T, img_feature_flatten_sel)
            
            match_type = "point2pixel"
            if match_type == "point2pixel":
                matched_pc = pc_sel
                sel_index = np.argmin(dist,axis=1)
                matched_img_xy = img_xy_flatten_sel[:,sel_index]
            elif match_type == "pixel2point":
                matched_img_xy = img_xy_flatten_sel
                sel_index = np.argmin(dist,axis=0)
                matched_pc = pc_sel[:,sel_index]
            elif match_type == "mutual_nn":
                min_values_row = np.min(dist, axis=1)[:, np.newaxis]
                mask_row = (dist <= min_values_row)

                min_values_col = np.min(dist, axis=0)[np.newaxis, :]
                mask_col = (dist <= min_values_col)

                mask = mask_row * mask_col
                sel_index_point, sel_index_pixel = np.where(mask)
                matched_pc = pc_sel[:,sel_index_point]
                matched_img_xy = img_xy_flatten_sel[:,sel_index_pixel]
            
            # visualize
            if step == 0:
                # input pcd
                origin_pc # 3 x num_point
                debug_point_cloud = o3d.geometry.PointCloud()
                debug_point_cloud.points = o3d.utility.Vector3dVector(origin_pc.T)
                o3d.io.write_point_cloud(f'z_vis_val/origin_input_pcd_{epoch}_{step}.ply', debug_point_cloud)
                
                debug_point_cloud = o3d.geometry.PointCloud()
                debug_point_cloud.points = o3d.utility.Vector3dVector(pc.T)
                o3d.io.write_point_cloud(f'z_vis_val/input_pcd_{epoch}_{step}.ply', debug_point_cloud)
                
                # input img
                debug_image = img[0].permute(1,2,0).cpu().numpy()
                debug_image = (debug_image * 255).astype(np.uint8)
                cv2.imwrite(f'z_vis_val/input_image_{epoch}_{step}.png', debug_image)
                
                # pred overlap points
                debug_point_cloud = o3d.geometry.PointCloud()
                debug_point_cloud.points = o3d.utility.Vector3dVector(pc_sel.T)
                o3d.io.write_point_cloud(f'z_vis_val/pred_overlap_origin_input_pcd_{epoch}_{step}.ply', debug_point_cloud)
                
                # matched pred overlap points
                debug_point_cloud = o3d.geometry.PointCloud()
                debug_point_cloud.points = o3d.utility.Vector3dVector(matched_pc.T)
                o3d.io.write_point_cloud(f'z_vis_val/matched_overlap_origin_input_pcd_{epoch}_{step}.ply', debug_point_cloud)
                
                # pred overlap xy
                debug_xy = (img_xy_flatten_sel * 4).astype(np.int32).T
                debug_image = cv2.imread(f'z_vis_val/input_image_{epoch}_{step}.png')
                for point in debug_xy:
                    cv2.circle(debug_image, (point[0], point[1]), 2, (0, 255, 0), -1)  # 绿色圆点
                cv2.imwrite(f'z_vis_val/pred_overlap_input_image_{epoch}_{step}.png', debug_image)
                
                # matched pred overlap xy
                debug_xy = (matched_img_xy * 4).astype(np.int32).T
                debug_image = cv2.imread(f'z_vis_val/input_image_{epoch}_{step}.png')
                for point in debug_xy:
                    cv2.circle(debug_image, (point[0], point[1]), 2, (0, 255, 0), -1)  # 绿色圆点
                cv2.imwrite(f'z_vis_val/matched_overlap_input_image_{epoch}_{step}.png', debug_image)
            else:
                None
            

            is_success,R,t,inliers=cv2.solvePnPRansac(matched_pc.T,matched_img_xy.T,K,useExtrinsicGuess=False,
                                                        iterationsCount=500,
                                                        reprojectionError=opt.dist_thres,
                                                        flags=cv2.SOLVEPNP_EPNP,
                                                        distCoeffs=None)
            R,_=cv2.Rodrigues(R)
            T_pred=np.eye(4)
            T_pred[0:3,0:3]=R
            T_pred[0:3,3:]=t
            t_diff,angles_diff=get_P_diff(T_pred,P)
            t_diff_set.append(t_diff)
            angles_diff_set.append(angles_diff)
            
    print(f"mean t_error = {np.mean(np.array(t_diff_set))}, mean r_error = {np.mean(np.array(angles_diff_set))}")
    print(f"median t_error = {np.median(np.array(t_diff_set))}, median r_error = {np.median(np.array(angles_diff_set))}")
    return np.mean(np.array(t_diff_set)),np.mean(np.array(angles_diff_set)), \
        np.median(np.array(t_diff_set)),np.median(np.array(angles_diff_set))

if __name__=='__main__':
    from carla import options_0905_baseline as options
    from carla.carla_pc_img_dataloader import make_carla_dataloader, carla_pc_img_dataset
    opt=options.Options()
    
    now = datetime.datetime.now()
    formatted_time = now.strftime("%m%d_%H%M")

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    logdir=os.path.join(opt.save_path, f'{formatted_time}_{opt.save_name}')
    try:
        os.makedirs(logdir)
    except:
        print('mkdir failue')

    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (logdir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # tensorboard logger
    tb_writer = SummaryWriter(log_dir=logdir)
    
    # create model
    model=CorrI2P(opt)
    model=model.cuda()

    # create train/val/test dataset & dataloader
    train_dataset, trainloader = make_carla_dataloader(mode='train', opt=opt)
    print('#training point clouds = %d' % len(train_dataset))
    valset, valloader = make_carla_dataloader(mode='val', opt=opt)
    print('#validating point clouds = %d' % len(valset))
    test_dataset, testloader = make_carla_dataloader(mode='test', opt=opt)
    print('#testing point clouds = %d' % len(test_dataset))
    
    current_lr=opt.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)

    global_step=0

    best_t_diff=1000
    best_r_diff=1000

    for epoch in range(1, 251):
        for step,data in enumerate(trainloader):
            global_step+=1
            model.train()
            optimizer.zero_grad()
            img=data['img'].cuda()                  #full size
            pc=data['pc'].cuda()
            pc_np_in_cam=data['pc_np_in_cam'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()      
            img_mask=data['img_mask'].cuda()        #1/4 size
            B=img_mask.size(0)
            pc_kpt_idx=data['pc_kpt_idx'].cuda()    #(B,512)
            pc_outline_idx=data['pc_outline_idx'].cuda()
            img_kpt_idx=data['img_kpt_idx'].cuda()
            img_outline_idx=data['img_outline_index'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()
            img_x=torch.linspace(0,img_mask.size(-1)-1,img_mask.size(-1)).view(1,-1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
            img_y=torch.linspace(0,img_mask.size(-2)-1,img_mask.size(-2)).view(-1,1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
            img_xy=torch.cat((img_x,img_y),dim=1)
            
            
            img_features,pc_features,img_score,pc_score=model(pc,intensity,sn,img,node_a,node_b)    #64 channels feature
            

            pc_features_inline=torch.gather(pc_features,index=pc_kpt_idx.unsqueeze(1).expand(B,pc_features.size(1),opt.num_kpt),dim=-1)
            pc_features_outline=torch.gather(pc_features,index=pc_outline_idx.unsqueeze(1).expand(B,pc_features.size(1),opt.num_kpt),dim=-1)
            pc_xyz_inline=torch.gather(pc,index=pc_kpt_idx.unsqueeze(1).expand(B,3,opt.num_kpt),dim=-1)
            # pc_xyz_inline=torch.gather(pc_np_in_cam,index=pc_kpt_idx.unsqueeze(1).expand(B,3,opt.num_kpt),dim=-1)
            pc_score_inline=torch.gather(pc_score,index=pc_kpt_idx.unsqueeze(1),dim=-1)
            pc_score_outline=torch.gather(pc_score,index=pc_outline_idx.unsqueeze(1),dim=-1)
            
            img_features_flatten=img_features.contiguous().view(img_features.size(0),img_features.size(1),-1)
            img_score_flatten=img_score.contiguous().view(img_score.size(0),img_score.size(1),-1)
            img_xy_flatten=img_xy.contiguous().view(img_features.size(0),2,-1)
            img_features_flatten_inline=torch.gather(img_features_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),opt.num_kpt),dim=-1)
            img_xy_flatten_inline=torch.gather(img_xy_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,2,opt.num_kpt),dim=-1)
            img_score_flatten_inline=torch.gather(img_score_flatten,index=img_kpt_idx.unsqueeze(1),dim=-1)
            img_features_flatten_outline=torch.gather(img_features_flatten,index=img_outline_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),opt.num_kpt),dim=-1)
            img_score_flatten_outline=torch.gather(img_score_flatten,index=img_outline_idx.unsqueeze(1),dim=-1)
            


            pc_xyz_projection=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc_xyz_inline)+P[:,0:3,3:]))
            #pc_xy_projection=torch.floor(pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]).float()
            pc_xy_projection=pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]

            correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=1))<=opt.dist_thres).float()
            

            loss_desc,dists=desc_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask,pos_margin=opt.pos_margin,neg_margin=opt.neg_margin, mode='contrastive', bin_score=model.bin_score)
            
            #loss_det=loss2.det_loss(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline,pc_score_outline.squeeze())
            loss_det=det_loss2(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline.squeeze(),pc_score_outline.squeeze(),dists,correspondence_mask)
            loss=loss_desc+loss_det*0.5
            #loss=loss_desc

            loss.backward()
            optimizer.step()
            
            #torch.cuda.empty_cache()

            if global_step == 50:
                os.system('nvidia-smi')
            if global_step%10==0:
                tb_writer.add_scalar('training_loss/desc_loss',
                                        loss_desc,
                                        global_step=global_step)
                tb_writer.add_scalar('training_loss/det_loss',
                                        loss_det*0.5,
                                        global_step=global_step)
                tb_writer.add_scalar('training_loss/loss',
                                        loss,
                                        global_step=global_step)
                tb_writer.add_scalar('learning_rate',
                                        current_lr,
                                        global_step=global_step)
                logger.info('%s-%d-%d, loss: %f, loss desc: %f, loss det: %f'%('train',epoch,global_step,loss.data.cpu().numpy(),loss_desc.data.cpu().numpy(),loss_det.data.cpu().numpy()))
            
        if epoch%5==0 and epoch>0:
            # eval on val loader (seen region)
            mean_t_diff,mean_r_diff,median_t_diff,median_r_diff=test_acc(model,valloader,epoch,opt)
            if median_t_diff<=best_t_diff:
                torch.save(model.state_dict(),os.path.join(logdir,'mode_best_t.t7'))
                best_t_diff=median_t_diff
            if median_r_diff<=best_r_diff:
                torch.save(model.state_dict(),os.path.join(logdir,'mode_best_r.t7'))
                best_r_diff=median_r_diff
            tb_writer.add_scalar('eval/median_t_diff (seen)',
                                    median_t_diff,
                                    global_step=global_step)
            tb_writer.add_scalar('eval/median_r_diff (seen)',
                                    median_r_diff,
                                    global_step=global_step)
            tb_writer.add_scalar('eval/mean_t_diff (seen)',
                                    mean_t_diff,
                                    global_step=global_step)
            tb_writer.add_scalar('eval/mean_r_diff (seen)',
                                    mean_r_diff,
                                    global_step=global_step)
            logger.info('eval on seen: %s-%d-%d, t_error: %f, r_error: %f'%('test',epoch,global_step,
                                                                            median_t_diff,median_r_diff))
            
            # eval on test lodaer (unseen region)
            mean_t_diff,mean_r_diff,median_t_diff,median_r_diff=test_acc(model,testloader,epoch,opt)
            tb_writer.add_scalar('eval/median_t_diff (unseen)',
                                    median_t_diff,
                                    global_step=global_step)
            tb_writer.add_scalar('eval/median_r_diff (unseen)',
                                    median_r_diff,
                                    global_step=global_step)
            tb_writer.add_scalar('eval/mean_t_diff (unseen)',
                                    mean_t_diff,
                                    global_step=global_step)
            tb_writer.add_scalar('eval/mean_r_diff (unseen)',
                                    mean_r_diff,
                                    global_step=global_step)
            logger.info('eval on unseen: %s-%d-%d, t_error: %f, r_error: %f'%('test',epoch,global_step,
                                                                              median_t_diff,median_r_diff))
            
            torch.save(model.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))
        
        if epoch%opt.lr_decay_step==0 and epoch>0:
            current_lr=current_lr*opt.lr_decay_scale
            if current_lr<opt.min_lr:
                current_lr=opt.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr']=current_lr
            logger.info('%s-%d-%d, updata lr, current lr is %f'%('train',epoch,global_step,current_lr))
            