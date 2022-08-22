#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import config
#from torchvision.utils import save_image
from matplotlib import pyplot as plt
import scipy.io as sio
from torchmetrics import PeakSignalNoiseRatio
import pytorch_ssim
import os

def saveImg3by3(v1, v2, v3, save_path):
    f, axarr = plt.subplots(3, 3, figsize=(15,15))
    axarr[0, 0].axis('off')
    axarr[0, 0].imshow(v1[0,0,:,:],"gray")
    axarr[0, 1].axis('off')
    axarr[0, 1].imshow(v1[1,0,:,:],"gray")
    axarr[0, 2].axis('off')
    axarr[0, 2].imshow(v1[2,0,:,:],"gray")
    
    axarr[1, 0].axis('off')   
    axarr[1, 0].imshow(v2[0,0,:,:],"gray")
    axarr[1, 1].axis('off')
    axarr[1, 1].imshow(v2[1,0,:,:],"gray")
    axarr[1, 2].axis('off')
    axarr[1, 2].imshow(v2[2,0,:,:],"gray")
    
    axarr[2, 0].axis('off')
    axarr[2, 0].imshow(v3[0,0,:,:],"gray")
    axarr[2, 1].axis('off')
    axarr[2, 1].imshow(v3[1,0,:,:],"gray")
    axarr[2, 2].axis('off')
    axarr[2, 2].imshow(v3[2,0,:,:],"gray")
    
    plt.savefig(save_path)
    plt.close(f)

def save_some_examples(gen, val_organMap_loader, val_target_loader, val_bodyMask_loader, epoch, writer, folder_imgs,folder_matfiles):
    x, y, bodyMask = next(iter(val_organMap_loader)), next(iter(val_target_loader)),next(iter(val_bodyMask_loader))
    x, y, bodyMask = x.to(config.DEVICE), y.to(config.DEVICE),bodyMask.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        x = x.double()
        x = x.unsqueeze(1)
        min_val = x.min(-1)[0].min(-1)[0]
        max_val = x.max(-1)[0].max(-1)[0]
        x = (x-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
        
        bodyMask = torch.abs(bodyMask.unsqueeze(1))
        x = x *bodyMask
        
        y_gen = gen(x)

        y = y.double()
        y = y.unsqueeze(1)
        
        min_val = y.min(-1)[0].min(-1)[0]
        max_val = y.max(-1)[0].max(-1)[0]
        y = (y-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
        y = y *bodyMask   
        
 
        saveImg3by3(y_gen.detach().cpu(), y.detach().cpu().numpy(), x.detach().cpu(),(folder_imgs+config.EXP_NAME+ f"/{epoch}.png"))

   
        #save y_gen matfile
        mdic_y_gen = {"y_gen": y_gen.detach().cpu().numpy()[0,0,:,:], "label": "Generated_image"}
        sio.savemat(folder_matfiles + config.EXP_NAME+f"/y_gen_{epoch}.mat", mdic_y_gen)
        #save y_input matfile
        mdic_x_input = {"x_input": x.detach().cpu().numpy()[0,0,:,:], "label": "Input_image"}
        sio.savemat(folder_matfiles + config.EXP_NAME+f"/x_input_{epoch}.mat", mdic_x_input)
        
        #save y_label matfile
        mdic_y_target = {"y_target": y.detach().cpu().numpy()[0,0,:,:], "label": "Target_image"}
        sio.savemat(folder_matfiles+ config.EXP_NAME+f"/y_label_{epoch}.mat", mdic_y_target)

        
        psnr = PeakSignalNoiseRatio()
        ssim_loss = pytorch_ssim.SSIM()
        psnr_value = psnr(y_gen.detach().cpu(),y.detach().cpu())
        ssim_value = ssim_loss(y_gen.detach().cpu(),y.detach().cpu())
        
        writer.add_scalar("psnr", psnr_value, epoch)
        writer.add_scalar("ssim", ssim_value, epoch)

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

