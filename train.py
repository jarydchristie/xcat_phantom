#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pytorch_ssim
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset_input, MapDataset_target, MapDataset_organMap, MapDataset_mask
from Generator_pix2pix import Generator
from Discriminator_pix2pix import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader_input, loader_organMap, loader_mask,
    loader_target, opt_disc, opt_gen, l1_loss, ssim_loss, bce, g_scaler, d_scaler,epoch
):
    loop_input = tqdm(loader_input, leave=True)

    for idx, x in enumerate(loop_input):
        x_input_simulatedImage = next(iter(loader_input)).to(config.DEVICE)
        x_input_simulatedImage = x_input_simulatedImage.double()
        x_input_simulatedImage = x_input_simulatedImage.unsqueeze(1)
        
        min_val = x_input_simulatedImage.min(-1)[0].min(-1)[0]
        max_val = x_input_simulatedImage.max(-1)[0].max(-1)[0]
        x_input_simulatedImage = (x_input_simulatedImage-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])

        
        x_input_organMap_Body = next(iter(loader_organMap)).to(config.DEVICE)
        x_input_organMap_Body = x_input_organMap_Body.double()
        x_input_organMap_Body = x_input_organMap_Body.unsqueeze(1)
        
        min_val = x_input_organMap_Body.min(-1)[0].min(-1)[0]
        max_val = x_input_organMap_Body.max(-1)[0].max(-1)[0]
        x_input_organMap_Body = (x_input_organMap_Body-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])

        
        x_input_mask = next(iter(loader_mask)).to(config.DEVICE)
        #x_input_mask = x_input_mask
        x_input_mask = torch.abs(x_input_mask.unsqueeze(1))
        
#        print(x_input_mask.shape)
        y_target = next(iter(loader_target)).to(config.DEVICE)
        y_target = y_target.double()
        y_target = y_target.unsqueeze(1)
        
        min_val = y_target.min(-1)[0].min(-1)[0]
        max_val = y_target.max(-1)[0].max(-1)[0]
        y_target = (y_target-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])

        
        x_input_organMap_GTV = x_input_organMap_Body * x_input_mask
        y_target_Body = y_target
        y_target_GTV = y_target * x_input_mask
        
        #num_slice = 1
        #plt.imshow(x_input_mask.cpu()[num_slice ,0,:,:],"gray")
        #plt.show()
        #plt.imshow(x_input_simulatedImage.cpu()[num_slice ,0,:,:],"gray")
        #plt.show()
        #plt.imshow(x_input_organMap_GTV.cpu()[num_slice ,0,:,:],"gray")
        #plt.show()
        #plt.imshow(x_input_organMap_Body.cpu()[num_slice ,0,:,:],"gray")
        #plt.show()
        #plt.imshow(y_target_Body.cpu()[num_slice ,0,:,:],"gray")
        #plt.show()
        #plt.imshow(y_target_GTV.cpu()[num_slice ,0,:,:],"gray")
        #plt.show()

        # Train Discriminator
        with torch.cuda.amp.autocast():
            
            y_fake_Body = gen(x_input_simulatedImage)
            
            #Body Discriminator_fake
            D_fake_Body = disc(x_input_organMap_Body, y_fake_Body.detach())
            D_fake_loss_Body = bce(D_fake_Body, torch.zeros_like(D_fake_Body))
            

            #Body Discriminator_real
            D_real_Body = disc(x_input_organMap_Body, y_target_Body)
            D_real_loss_Body = bce(D_real_Body, torch.ones_like(D_real_Body))
            
            #Body Discriminator loss
            D_loss_Body = (D_real_loss_Body + D_fake_loss_Body) / 2
            
            
            #GTV Discriminator_fake
            D_fake_GTV = disc(x_input_organMap_GTV,(y_fake_Body.detach() * x_input_mask))
            D_fake_loss_GTV = bce(D_fake_GTV, torch.zeros_like(D_fake_GTV))
            
            #GTV Discriminator_real
            D_real_GTV = disc(x_input_organMap_GTV, y_target_GTV)
            D_real_loss_GTV = bce(D_real_GTV, torch.ones_like(D_real_GTV))
            
            #GTV Discriminator loss
            D_loss_GTV = (D_real_loss_GTV + D_fake_loss_GTV) / 2
            
            
            #Total Discriminator loss
            D_loss = D_loss_Body + D_loss_GTV 
#            print(y_fake_Body.shape)
#            plt.imshow(y_fake_Body.detach().cpu()[0,0,:,:],"gray")
#            plt.show()
#            mdic = {"y_fake_train": y_fake_Body.detach().cpu().numpy()[0,0,:,:], "label": "Generated_image"}
#            sio.savemat("y_fake_train.mat", mdic)
#            print("D_loss_Body",D_loss_Body)
#            print("D_loss_GTV", D_loss_GTV)

        writer.add_scalar("D_loss/train", D_loss, epoch)
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            
            D_fake_Body = disc(x_input_organMap_Body, y_fake_Body) 
            D_fake_GTV = disc(x_input_organMap_GTV,(y_fake_Body * x_input_mask))
            D_3rd_term = disc(x_input_simulatedImage,y_target)
            
            G_loss_Body =bce(D_fake_Body, torch.ones_like(D_fake_Body))
            G_loss_GTV =bce(D_fake_GTV, torch.ones_like(D_fake_GTV))
            G_3rd_term_loss = bce(D_3rd_term, torch.ones_like(D_3rd_term))
            
            SSIM_Body = (1-ssim_loss(y_fake_Body.detach(), y_target_Body)) * config.SSIM_MU
#            plt.imshow((y_fake_Body.detach().cpu())[0,0,:,:],"gray")
#            plt.show()             
#            plt.imshow((y_fake_Body.detach().cpu()* x_input_mask.detach().cpu())[0,0,:,:],"gray")
#            plt.show()  
#            mdic = {"xmask": ((y_fake_Body.detach().cpu()*x_input_mask.detach().cpu())[0,0,:,:].numpy()), "label": "Generated_image"}
#            sio.savemat("xmask.mat", mdic)            
#            plt.imshow(y_target_GTV.detach().cpu()[0,0,:,:],"gray")
#            plt.show() 
#            mdic = {"ymask": (y_target_GTV.detach().cpu()[0,0,:,:].numpy()), "label": "Generated_image"}
#            sio.savemat("ymask.mat", mdic)              
            SSIM_GTV = (1-ssim_loss((y_fake_Body.detach() * x_input_mask), y_target_GTV)) * config.SSIM_V

            L1 = l1_loss(y_fake_Body, y_target_Body) * config.L1_LAMBDA
            

            
            G_loss = (L1  + G_loss_Body + SSIM_Body + G_loss_Body + SSIM_GTV + G_loss_GTV + G_3rd_term_loss)/7
#            print("L1",L1)
#            print("SSIM_Body",SSIM_Body)
#            print("SSIM_GTV",SSIM_GTV)
#            print("G_loss_Body",G_loss_Body)
#            print("G_loss_GTV",G_loss_GTV)
#            print("G_3rd_term_loss",G_3rd_term_loss)
        writer.add_scalar("G_loss/train", G_loss, epoch)
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop_input.set_postfix(
                D_real_Body=torch.sigmoid(D_real_Body).mean().item(),
                D_fake_Body=torch.sigmoid(D_fake_Body).mean().item(),
                D_real_GTV=torch.sigmoid(D_real_GTV).mean().item(),
                D_fake_GTV=torch.sigmoid(D_fake_GTV).mean().item(),                
            )


def main():
    disc = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    disc=disc.double()
    gen = Generator(in_channels=config.CHANNELS_IMG, features=32).to(config.DEVICE)
    gen=gen.double()
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    SSIM_LOSS = pytorch_ssim.SSIM()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_input_dataset = MapDataset_input(input_root_dir=config.TRAIN_INPUT_DIR)
    train_input_dataset.input_list_files = train_input_dataset.input_list_files[0:8]#######
    train_input_loader = DataLoader(train_input_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=False,num_workers=config.NUM_WORKERS)
    train_organMap_dataset = MapDataset_organMap(organMap_root_dir=config.TRAIN_ORGANMAP_DIR)
    train_organMap_dataset.organMap_list_files = train_organMap_dataset.organMap_list_files[0:8]######    
    train_organMap_loader = DataLoader(train_organMap_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=False,num_workers=config.NUM_WORKERS)
    train_mask_dataset = MapDataset_mask(mask_root_dir=config.TRAIN_MASK_DIR)
    train_mask_dataset.mask_list_files = train_mask_dataset.mask_list_files[0:8]######      
    train_mask_loader = DataLoader(train_mask_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=False,num_workers=config.NUM_WORKERS)    
    train_target_dataset = MapDataset_target(target_root_dir=config.TRAIN_TARGET_DIR)
    train_target_dataset.target_list_files = train_target_dataset.target_list_files[0:8]######
    train_target_loader = DataLoader(train_target_dataset, batch_size=config.BATCH_SIZE,
                                     shuffle=False,num_workers=config.NUM_WORKERS)
    
 



    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    g_cpu = torch.Generator()
    g_cpu_other = torch.Generator()  
    g_cpu.set_state(g_cpu_other.get_state())
    
    
#    val_input_dataset = MapDataset_input(input_root_dir=config.VAL_INPUT_DIR)
#    val_input_loader = DataLoader(val_input_dataset, batch_size=config.BATCH_SIZE,
#                                    shuffle=False,num_workers=config.NUM_WORKERS)
    val_organMap_dataset = MapDataset_organMap(organMap_root_dir=config.VAL_ORGANMAP_DIR)
    val_organMap_loader = DataLoader(val_organMap_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True,generator = g_cpu,num_workers=config.NUM_WORKERS)
#    val_mask_dataset = MapDataset_mask(mask_root_dir=config.VAL_MASK_DIR)
#    val_mask_loader = DataLoader(val_mask_dataset, batch_size=config.BATCH_SIZE,
#                                    shuffle=False,num_workers=config.NUM_WORKERS)    
    val_target_dataset = MapDataset_target(target_root_dir=config.VAL_TARGET_DIR)
    val_target_loader = DataLoader(val_target_dataset, batch_size=config.BATCH_SIZE,
                                     shuffle=True,generator = g_cpu_other,num_workers=config.NUM_WORKERS) 

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_input_loader, train_organMap_loader, train_mask_loader, train_target_loader,
            opt_disc, opt_gen, L1_LOSS, SSIM_LOSS, BCE, g_scaler, d_scaler,epoch,
        )
        writer.flush()

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_organMap_loader, val_target_loader, epoch, folder="evaluation")
    


if __name__ == "__main__":
    main()


# In[ ]:


writer.close()


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir=runs')

