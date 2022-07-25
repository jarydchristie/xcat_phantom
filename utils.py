#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import config
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import scipy.io as sio

def save_some_examples(gen, val_organMap_loader, val_target_loader, epoch, folder):
    x, y = next(iter(val_organMap_loader)), next(iter(val_target_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        x = x.unsqueeze(1)
        min_val = x.min(-1)[0].min(-1)[0]
        max_val = x.max(-1)[0].max(-1)[0]
        x = (x-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
        y_fake = gen(x)

        y_gen = y_fake# * 0.5 + 0.5  # remove normalization#
        y = y.unsqueeze(1)
        
        plt.imsave(folder + f"/y_gen_{epoch}.png", y_gen.detach().cpu()[0,0,:,:], cmap='gray')
   
        #save matfile
        mdic_y_gen = {"y_gen": y_gen.detach().cpu().numpy()[0,0,:,:], "label": "Generated_image"}
        sio.savemat(folder + f"/y_gen_{epoch}.mat", mdic_y_gen)
        
        plt.imsave(folder + f"/input_{epoch}.png", x.detach().cpu()[0,0,:,:], cmap='gray')# * 0.5 + 0.5
        
        mdic_y_target = {"y_target": y.detach().cpu().numpy()[0,0,:,:], "label": "Generated_image"}
        sio.savemat(folder + f"/y_gen_{epoch}.mat", mdic_y_target)        
        plt.imsave(folder + f"/label_{epoch}.png", y.detach().cpu()[0,0,:,:], cmap='gray')# * 0.5 + 0.5
        
        #plt.imshow((y.detach().cpu())[0,0,:,:],"gray")
        #plt.show()
        #plt.imshow((y_gen.detach().cpu())[0,0,:,:],"gray")
        #plt.show() 
        #if epoch == 1:

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

