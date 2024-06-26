# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import torch
import mvtech
import torch.nn.functional as F
import os
import numpy as np
import pytorch_ssim
from einops import rearrange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import mdn1
from VT_AE import VT_AE as ae
from utility_fun import *

object = "hazelnut"
patch_size = 64

ssim_loss = pytorch_ssim.SSIM() # SSIM Loss

#Dataset
data = mvtech.Mvtec(batch_size=1,product=object, root="/home/evry/Desktop/master-degree/repositories/two-stage-coarse-to-fine-image-anomaly-segmentation-and-detection-model/processed_data")

# Model declaration
model = ae(train=False).cuda()
G_estimate= mdn1.MDN().cuda()

# Loading weights
model.load_state_dict(torch.load(f'./saved_model/VT_AE_Mvtech_{object}'+'.pt'))
G_estimate.load_state_dict(torch.load(f'./saved_model/G_estimate_Mvtech_{object}'+'.pt'))

#put model to eval
model.eval()
G_estimate.eval()


#### testing #####
loader = [None,data.test_norm_loader,data.test_anom_loader]

t_loss_norm =[]
t_loss_anom =[]

def Thresholding(data_load = loader[1:], upsample = 1, thres_type = 0, fpr_thres = 0.3):
    '''
    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is data.train_loader.
    upsample : INT, optional
        DESCRIPTION. 0 - NearestUpsample2d; 1- BilinearUpsampling.
    thres_type : INT, optional
        DESCRIPTION. 0 - 30% of fpr reached; 1 - thresholding using best F1 score
    fpr_thres : FLOAT, Optional
        DESCRIPTION. False Positive Rate threshold value. Default is 0.3

    Returns
    -------
    Threshold: Threshold value

    '''
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []

    for data in data_load:
        for image, mask in data:
            if image.shape[1] == 4:
                image = image[:, :3, :, :]
                
            if mask.shape[1] == 4:
                mask = mask[:, :3, :, :]

            if image.size(1)==1:
                image = torch.stack([image,image,image]).squeeze(2).permute(1,0,2,3)
            vector, reconstructions = model(image.cuda())
            pi, mu, sigma = G_estimate(vector)
            
            #Loss calculations
            loss1 = F.mse_loss(reconstructions,image.cuda(), reduction='mean') #Rec Loss
            loss2 = -ssim_loss(image.cuda(), reconstructions) #SSIM loss for structural similarity
            loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True) #MDN loss for gaussian approximation
            loss = loss1 + loss2 + loss3.sum()       #Total loss
            norm_loss_t.append(loss3.detach().cpu().numpy())
                
            if upsample==0 :
                #Mask patch
                mask_patch = rearrange(mask.squeeze(0).squeeze(0), '(h p1) (w p2) -> (h w) p1 p2', p1 = patch_size, p2 = patch_size)
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1),0.)
                mask_score_t.append(mask_patch_score) # Storing all masks
                norm_score = norm_loss_t[-1]
                normalised_score_t.append(norm_score)# Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(mask.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks
                m = torch.nn.UpsamplingBilinear2d((512,512))
                norm_score = norm_loss_t[-1].reshape(-1,1,512//patch_size,512//patch_size)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map , type =0) # add normalization here for the testing
                normalised_score_t.append(score_map) # Storing all score maps      
                
            # break
                
                
    print(f"scores shape {normalised_score_t[0].shape} size: {len(normalised_score_t)}")
    scores = np.asarray(normalised_score_t).flatten()
    print(f"masks shape {mask_score_t[0].shape} size: {len(mask_score_t)}")
    masks = np.asarray(mask_score_t).flatten()
    # 262144 (works)
    # 28835840(works) 
    # 102498304 (dont)
    if thres_type == 0 :
        fpr, tpr, _ = roc_curve(masks, scores)
        fp3 = np.where(fpr<=fpr_thres)
        threshold = _[fp3[-1][-1]]
    elif thres_type == 1:
        precision, recall, thresholds = precision_recall_curve(masks, scores)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)] 
    return threshold
    

def Patch_Overlap_Score(threshold, data_load = loader[1:], upsample =1):
    
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []
    loss1_tn = []
    loss2_tn = []
    loss3_tn = []
    loss1_ta = []
    loss2_ta = []
    loss3_ta = []
    
    score_tn = []
    score_ta = []
    

    for data_loader_idx, data in enumerate(data_load):
        total_loss_all = []
        
        anomaly_data = data_loader_idx == 1
        
        images_path = f"./runs/test/images/{object}/{'anomaly' if anomaly_data else 'norm'}"
        
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        
        for sample_idx, (image, ground_truth) in enumerate(data):
            if image.size(1)==1:
                image = torch.stack([image,image,image]).squeeze(2).permute(1,0,2,3)
            vector, reconstructions = model(image.cuda())
            pi, mu, sigma = G_estimate(vector)
           
            #Loss calculations
            loss1 = F.mse_loss(reconstructions,image.cuda(), reduction='mean') #Rec Loss
            loss2 = -ssim_loss(image.cuda(), reconstructions) #SSIM loss for structural similarity
            loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True) #MDN loss for gaussian approximation
            loss = loss1 -loss2 + loss3.max()       #Total loss
            norm_loss_t.append(loss3.detach().cpu().numpy())
            total_loss_all.append(loss.detach().cpu().numpy())
            
            if data_loader_idx == 0 :
                loss1_tn.append(loss1.detach().cpu().numpy())
                loss2_tn.append(loss2.detach().cpu().numpy())
                loss3_tn.append(loss3.sum().detach().cpu().numpy())
            if data_loader_idx == 1:
                loss1_ta.append(loss1.detach().cpu().numpy())
                loss2_ta.append(loss2.detach().cpu().numpy())
                loss3_ta.append(loss3.sum().detach().cpu().numpy())
                
            if upsample==0 :
                #Mask patch
                mask_patch = rearrange(ground_truth.squeeze(0).squeeze(0), '(h p1) (w p2) -> (h w) p1 p2', p1 = patch_size, p2 = patch_size)
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1),0.)
                mask_score_t.append(mask_patch_score) # Storing all masks
                norm_score = Binarization(norm_loss_t[-1], threshold)
                m = torch.nn.UpsamplingNearest2d((512,512))
                score_map = m(torch.tensor(norm_score.reshape(-1,1,512//patch_size,512//patch_size)))
               
                
                normalised_score_t.append(norm_score)# Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(ground_truth.squeeze(0).squeeze(0).cpu().numpy()) # Storing all masks
                
                m = torch.nn.UpsamplingBilinear2d((512,512))
                norm_score = norm_loss_t[-1].reshape(-1,1,512//patch_size,512//patch_size)
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map , type =0) 

                   
                normalised_score_t.append(score_map) # Storing all score maps
                
            ## Plotting
            if sample_idx == 0:
                print(f"Img shape: {image.shape} gt shape: {ground_truth.shape}, score shape: {score_map[0][0].shape}")
            save_figure(image,ground_truth[0],score_map[0][0], f"{images_path}/{sample_idx}.png")
            
            if data_loader_idx == 0:
                score_tn.append(score_map.max())
            if data_loader_idx ==1:
                score_ta.append(score_map.max())
                
                
        if data_loader_idx == 0 :
            t_loss_all_normal = total_loss_all
        if data_loader_idx == 1:
            t_loss_all_anomaly = total_loss_all
        
    ## PRO Score            
    scores = np.asarray(normalised_score_t).flatten()
    masks = np.asarray(mask_score_t).flatten()
    PRO_score = roc_auc_score(masks, scores)
    
    ## Image Anomaly Classification Score (AUC)
    roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
    roc_targets = np.concatenate((np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly))))
    AUC_Score_total = roc_auc_score(roc_targets, roc_data)
    
    # AUC Precision Recall Curve
    precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
    AUC_PR = auc(recall, precision)

    
    return PRO_score, AUC_Score_total, AUC_PR

if __name__=="__main__":
    
    thres = Thresholding()
    PRO, AUC, AUC_PR = Patch_Overlap_Score(threshold=thres)

    print(f'PRO Score: {PRO} \nAUC Total: {AUC} \nPR_AUC Total: {AUC_PR}')

