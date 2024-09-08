import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import argparse
from network import CorrI2P
from kitti_pc_img_dataloader import kitti_pc_img_dataset
#from loss2 import kpt_loss, kpt_loss2, eval_recall
import datetime
import logging
import math
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation
import cv2

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    
    return t_diff,angles_diff
                
if __name__=='__main__':
    from carla import options as options
    from carla.carla_pc_img_dataloader import make_carla_dataloader, carla_pc_img_dataset
    
    opt=options.Options()
    test_dataset, testloader = make_carla_dataloader(mode='val', opt=opt)
    print('#testing point clouds = %d' % len(test_dataset))
    
    model_path = "runs_carla/0905_1529_baseline/mode_epoch_15.t7"
    save_path = './eval_result/demo15'
    
    model=CorrI2P(opt)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict, strict=True)
    
    model=model.cuda()
    
    try:
        os.mkdir(save_path)
    except:
        pass
    with torch.no_grad():
        for step,data in tqdm.tqdm(enumerate(testloader),total=len(testloader)):
            model.eval()
            img=data['img'].cuda()
            pc=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()
            img_mask=data['img_mask'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()
            pc_feature=torch.cat((intensity,sn),dim=1)
            img_feature,pc_feature,img_score,pc_score=model(pc,intensity,sn,img,node_a,node_b)
            
            np.save(os.path.join(save_path,'img_%d.npy'%(step)),img.cpu().numpy())
            np.save(os.path.join(save_path,'pc_%d.npy'%(step)),pc.cpu().numpy())
            np.save(os.path.join(save_path,'pc_score_%d.npy'%(step)),pc_score.data.cpu().numpy())
            np.save(os.path.join(save_path,'pc_mask_%d.npy'%(step)),pc_mask.data.cpu().numpy())
            np.save(os.path.join(save_path,'K_%d.npy'%(step)),K.data.cpu().numpy())
            np.save(os.path.join(save_path,'img_mask_%d.npy'%(step)),img_mask.data.cpu().numpy())
            np.save(os.path.join(save_path,'img_score_%d.npy'%(step)),img_score.data.cpu().numpy())
            np.save(os.path.join(save_path,'img_feature_%d.npy'%(step)),img_feature.data.cpu().numpy())
            np.save(os.path.join(save_path,'pc_feature_%d.npy'%(step)),pc_feature.data.cpu().numpy())
            np.save(os.path.join(save_path,'P_%d.npy'%(step)),P.data.cpu().numpy())
            

            
            debug_eval = False
            if debug_eval:
                # eval
                img_score=img_score[0].data.cpu().numpy()
                pc_score=pc_score[0].data.cpu().numpy()
                img_feature=img_feature[0].data.cpu().numpy()
                pc_feature=pc_feature[0].data.cpu().numpy()
                pc=pc[0].cpu().numpy()
                # P=P_set[i]
                K=K[0].cpu().numpy()
                P=P[0].cpu().numpy()
                
                img_x=np.linspace(0,np.shape(img_feature)[-1]-1,np.shape(img_feature)[-1]).reshape(1,-1).repeat(np.shape(img_feature)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
                img_y=np.linspace(0,np.shape(img_feature)[-2]-1,np.shape(img_feature)[-2]).reshape(-1,1).repeat(np.shape(img_feature)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
                img_xy=np.concatenate((img_x,img_y),axis=0)

                #print(img_xy[:,22,1])

                img_xy_flatten=img_xy.reshape(2,-1)
                img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
                img_score_flatten=img_score.squeeze().reshape(-1)
                
                img_xy_flatten_sel=img_xy_flatten[:,img_score_flatten>0.9]
                img_feature_flatten_sel=img_feature_flatten[:,img_score_flatten>0.9]
                img_score_flatten_sel=img_score_flatten[img_score_flatten>0.9]
                
                pc_sel=pc[:,pc_score.squeeze()>0.9]
                pc_feature_sel=pc_feature[:,pc_score.squeeze()>0.9]
                pc_score_sel=pc_score.squeeze()[pc_score.squeeze()>0.9]
                
                dist=1-np.dot(pc_feature_sel.T, img_feature_flatten_sel)
                sel_index=np.argsort(dist,axis=1)[:,0]
                img_xy_pc=img_xy_flatten_sel[:,sel_index]

                try:
                    is_success,R,t,inliers=cv2.solvePnPRansac(pc_sel.T,img_xy_pc.T,K,useExtrinsicGuess=False,
                                                                iterationsCount=500,
                                                                reprojectionError=1,
                                                                flags=cv2.SOLVEPNP_EPNP,
                                                                distCoeffs=None)
                except:
                    print('has problem!')
                    print('pc shape',pc_sel.shape,'img shape',img_xy_pc.shape)
                    assert False
                R,_=cv2.Rodrigues(R)
                '''print(R)
                print(t)
                print(P)
                print(is_success)'''
                T_pred=np.eye(4)
                T_pred[0:3,0:3]=R
                T_pred[0:3,3:]=t
                #print(T_pred)

                # P = np.eye(4)
                t_diff,angles_diff=get_P_diff(T_pred,P)
                print(t_diff)
                print(angles_diff)
            