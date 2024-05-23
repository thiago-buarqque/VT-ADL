# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import math
import torch
import mvtech
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pytorch_ssim
import mdn1
from VT_AE import VT_AE as ae
import argparse
import logging
from progressbar import Bar, DynamicMessage, ProgressBar, ETA
from time import gmtime, strftime
from torchsummary import summary

## Argparse declaration ##

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--product", required=True,default = 'hazelnut',help="product from the dataset MvTec or BTAD", action='store', type=str,nargs='+')
ap.add_argument("-e", "--epochs", required=False, default= 1000, help="Number of epochs to train")
ap.add_argument("-lr", "--learning_rate", required=False, default= 0.0001, help="learning rate")
ap.add_argument("-ps","--patch_size", required=False, default=64, help="Patch size of the images")
ap.add_argument("-b", "--batch_size", required=False, default=16, help= "batch size")
args = vars(ap.parse_args())

# Setup logging

# prdt = args["product"]
for class_name in args["product"][0].split(","):   
    # Create logger
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create file handler which logs even debug messages
    # fh = logging.FileHandler(f'./logs/{class_name}_{strftime("%Y-%m-%d-%H:%M:%S", gmtime())}.log', mode='w')  # Open in write mode, which will create the file if it does not exist
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))  # Include timestamp in the log

    # # Add file handler to the logger
    # logger.addHandler(fh)
    
    writer = SummaryWriter()

    epochs = int(args["epochs"])
    minloss = 1e10
    ep =0
    ssim_loss = pytorch_ssim.SSIM() # SSIM Loss

    #Dataset
    data = mvtech.Mvtec(int(args["batch_size"]),product=class_name)

    # Model declaration
    model = ae(patch_size=int(args["patch_size"]),train=True).cuda()
    G_estimate= mdn1.MDN().cuda()

    ### put model to train ## 
    #(The two models are trained as a separate module so that it would be easy to use as an independent module in different scenarios)
    model.train()
    G_estimate.train()
    
    # print(summary(model, (1, 512, 512)))

    #Optimiser Declaration
    optimizer = Adam(list(model.parameters())+list(G_estimate.parameters()), lr=args["learning_rate"], weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[epochs*0.2,epochs*0.4,epochs*0.6, epochs*0.8],gamma=0.1, last_epoch=-1)

    ############## TRAIN #####################
    # torch.autograd.set_detect_anomaly(True) #uncomment if you want to track an error

    print('\nNetwork training started for {class_name}...')
    for epoch in range(epochs):
        epoch_losses = []
        
        widgets = [
                    DynamicMessage('epoch'),
                    Bar(marker='=', left='[', right=']'),
            ' ',  ETA(),
        ]
        
        with ProgressBar(widgets=widgets, max_value=data.train_loader.__len__()) as progress_bar:
            for sample_i, (image, mask) in enumerate(data.train_loader):
                if image.size(1)==1:
                    image = torch.stack([image,image,image]).squeeze(2).permute(1,0,2,3)

                model.zero_grad()

                # vector,pi, mu, sigma, reconstructions = model(j.cuda())
                vector, reconstructions = model(image.cuda())
                
                pi, mu, sigma = G_estimate(vector)
                
                #Loss calculations
                mse = F.mse_loss(reconstructions,image.cuda(), reduction='mean') #Rec Loss
                ssim = -ssim_loss(image.cuda(), reconstructions) #SSIM loss for structural similarity
                mdn = mdn1.mdn_loss_function(vector,mu,sigma,pi) #MDN loss for gaussian approximation
                
                # print(f' loss3  : {loss3.item()}')
                loss = 5 * mse + 0.5 * ssim + mdn       #Total loss
                
                epoch_losses.append(loss.item())   #storing all batch losses to calculate mean epoch loss
                
                # Tensorboard definitions
                writer.add_scalar('MSE-loss', mse.item(), epoch)
                writer.add_scalar('SSIM loss', ssim.item(), epoch)
                writer.add_scalar('Gaussian loss', mdn.item(), epoch)
                
                # if len(vector) >= 2:
                #     writer.add_histogram('Vectors', vector)
                
                ## Uncomment below to store the distributions of pi, var and mean ##        
                # writer.add_histogram('Pi', pi)
                # writer.add_histogram('Variance', sigma)
                # writer.add_histogram('Mean', mu)

                #Optimiser step
                loss.backward()
                optimizer.step()
                
                log = f"{epoch} ({sample_i}/{data.train_loader.__len__()}) Class {class_name} | MSE: {mse:.2f} | SSIM: {ssim:.2f} | MDN: {mdn:.2f} | Losses 'sum'/mean/min: {loss:.2f} / {np.mean(epoch_losses):.2f} / {minloss:.2f}"
                
                progress_bar.update(
                        sample_i,
                        epoch=log)
                
                # logging.info(log)
        
            progress_bar.finish()

        loss_mean = np.mean(epoch_losses)

        #Tensorboard definitions for the mean epoch values
        writer.add_image('Original Image',utils.make_grid(image),epoch,dataformats = 'CHW')
        writer.add_image('Reconstructed Image',utils.make_grid(reconstructions),epoch,dataformats = 'CHW')
        writer.add_scalar('Mean Epoch loss', loss_mean, epoch)
        # print(f'Mean Epoch {i} loss: {loss_mean}')
        # print(f'Min loss epoch: {ep} with min loss: {minloss}')
            
        writer.close()
        
        if loss_mean is np.nan or loss_mean is math.nan or str(loss_mean) == 'nan':
            log = f"Loss mean is nan: {epoch_losses}"
            
            # logging.info(log)
            
            # print(log)
            break
        
        # Saving the best model
        if loss_mean <= minloss:
            minloss = loss_mean
            ep = epoch
            os.makedirs('./saved_model', exist_ok=True)
            torch.save(model.state_dict(), f'./saved_model/VT_AE_Mvtech_{class_name}'+'.pt')
            torch.save(G_estimate.state_dict(), f'./saved_model/G_estimate_Mvtech_{class_name}'+'.pt')
            

    '''
    Full forms:
    GN - gaussian Noise
    LD = Linear Decoder
    DR - Dynamic Routing
    Gn = No of gaussian for the estimation of density, with n as the number
    Pn = Pacth with n is dim of patch
    SS - trained with ssim loss


    '''