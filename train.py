#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pytorch_ssim
from utils import save_checkpoint, load_checkpoint, save_some_examples, saveImg3by3
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset_input, MapDataset_target, MapDataset_organMap, MapDataset_mask, MapDataset_bodyMask
from Generator_pix2pix import Generator
from Discriminator_pix2pix import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
#from torchvision.utils import save_image
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.backends.cudnn.benchmark = True


def train_fn(
    disc_Body,disc_GTV, gen, loader_input, loader_organMap, loader_mask, loader_bodyMask,
    loader_target, opt_disc_Body,opt_disc_GTV, opt_gen, l1_loss,l2_loss, ssim_loss, bce, g_scaler, 
    d_scaler_Body,d_scaler_GTV,epoch
):
    loop_input = tqdm(loader_input, leave=True)

    for idx, x in enumerate(loop_input):
        #Body mask load
        x_input_bodyMask = next(iter(loader_bodyMask)).to(config.DEVICE)
        x_input_bodyMask = torch.abs(x_input_bodyMask.unsqueeze(1))
        
        #Simulated image load   
        x_input_simulatedImage = next(iter(loader_input)).to(config.DEVICE)
        x_input_simulatedImage = x_input_simulatedImage.double()
        x_input_simulatedImage = x_input_simulatedImage.unsqueeze(1)
        
        min_val = x_input_simulatedImage.min(-1)[0].min(-1)[0]
        max_val = x_input_simulatedImage.max(-1)[0].max(-1)[0]
        x_input_simulatedImage = (x_input_simulatedImage-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
        x_input_simulatedImage = x_input_simulatedImage * x_input_bodyMask

        
        #Organ Map image load
        x_input_organMap_Body = next(iter(loader_organMap)).to(config.DEVICE)
        x_input_organMap_Body = x_input_organMap_Body.double()
        x_input_organMap_Body = x_input_organMap_Body.unsqueeze(1)
        
        min_val = x_input_organMap_Body.min(-1)[0].min(-1)[0]
        max_val = x_input_organMap_Body.max(-1)[0].max(-1)[0]
        x_input_organMap_Body = (x_input_organMap_Body-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
        x_input_organMap_Body = x_input_organMap_Body * x_input_bodyMask

        # GTV mask load
        x_input_GTVmask = next(iter(loader_mask)).to(config.DEVICE)
        x_input_GTVmask = torch.abs(x_input_GTVmask.unsqueeze(1))
        
        #Body Target load
        y_target = next(iter(loader_target)).to(config.DEVICE)
        y_target = y_target.double()
        y_target = y_target.unsqueeze(1)
        
        min_val = y_target.min(-1)[0].min(-1)[0]
        max_val = y_target.max(-1)[0].max(-1)[0]
        y_target = (y_target-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])
        y_target_Body = y_target * x_input_bodyMask

        
        #GTV Target load
        x_input_organMap_GTV = x_input_organMap_Body * x_input_GTVmask
        y_target_GTV = y_target * x_input_GTVmask
        
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
            
            y_fake_Body = gen(x_input_organMap_Body)
            
            #Body Discriminator_fake
            D_fake_Body = disc_Body(x_input_organMap_Body, y_fake_Body.detach())
            D_fake_loss_Body = l2_loss(D_fake_Body, torch.zeros_like(D_fake_Body))
            

            #Body Discriminator_real
            D_real_Body = disc_Body(x_input_organMap_Body, y_target_Body)
            D_real_loss_Body = l2_loss(D_real_Body, torch.ones_like(D_real_Body))
            
            #Body Discriminator loss
            D_loss_Body = (D_real_loss_Body + D_fake_loss_Body) / 2
            
            
            #GTV Discriminator_fake
            D_fake_GTV = disc_GTV(x_input_organMap_GTV,(y_fake_Body.detach() * x_input_GTVmask))
            D_fake_loss_GTV = l2_loss(D_fake_GTV, torch.zeros_like(D_fake_GTV))
            
            #GTV Discriminator_real
            D_real_GTV = disc_GTV(x_input_organMap_GTV, y_target_GTV)
            D_real_loss_GTV = l2_loss(D_real_GTV, torch.ones_like(D_real_GTV))
            
            #GTV Discriminator loss
            D_loss_GTV = (D_real_loss_GTV + D_fake_loss_GTV) / 2
        
        
        if epoch % 5 == 0:
            saveImg3by3(y_fake_Body.detach().cpu(), y_target_Body.detach().cpu().numpy(), x_input_organMap_Body.detach().cpu(),("evaluationTrials/l2/images/"+config.EXP_NAME+ f"/{epoch}train.png"))


        writer.add_scalar("D_loss_Body/train", D_loss_Body, epoch)
        disc_Body.zero_grad()
        d_scaler_Body.scale(D_loss_Body).backward(retain_graph=True)
        d_scaler_Body.step(opt_disc_Body)
        d_scaler_Body.update()
        
        writer.add_scalar("D_loss_GTV/train", D_loss_GTV, epoch)
        disc_GTV.zero_grad()
        d_scaler_GTV.scale(D_loss_GTV).backward(retain_graph=True)
        d_scaler_GTV.step(opt_disc_GTV)
        d_scaler_GTV.update()

        # Train generator
        with torch.cuda.amp.autocast():
            
            D_fake_Body = disc_Body(x_input_organMap_Body, y_fake_Body) 
            D_fake_GTV = disc_GTV(x_input_organMap_GTV,(y_fake_Body* x_input_GTVmask))
            D_3rd_term = disc_Body(x_input_organMap_Body,y_target_Body)
            
            G_loss_Body =l2_loss(D_fake_Body, torch.ones_like(D_fake_Body))
            G_loss_GTV =l2_loss(D_fake_GTV, torch.ones_like(D_fake_GTV))
            G_3rd_term_loss = l2_loss(D_3rd_term, torch.ones_like(D_3rd_term))
            
            SSIM_Body = (1-ssim_loss(y_fake_Body, y_target_Body)) * config.SSIM_MU            
            SSIM_GTV = (1-ssim_loss((y_fake_Body * x_input_GTVmask), y_target_GTV)) * config.SSIM_V

            L1 = l1_loss(y_fake_Body, y_target_Body) * config.L1_LAMBDA
            

            G_loss = (L1  + SSIM_Body + G_loss_Body + SSIM_GTV + G_loss_GTV+G_3rd_term_loss)          

        writer.add_scalar("L1/train", L1, epoch)
        writer.add_scalar("SSIM_Body/train", SSIM_Body, epoch)
        writer.add_scalar("SSIM_GTV/train", SSIM_GTV, epoch)
        writer.add_scalar("G_loss_Body/train", G_loss_Body, epoch)
        writer.add_scalar("G_loss_GTV/train", G_loss_GTV, epoch)       
        writer.add_scalar("G_3rd_term_loss/train", G_3rd_term_loss, epoch)
        writer.add_scalar("G_loss/train", G_loss, epoch)
          
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

#        if idx % 10 == 0:
#            loop_input.set_postfix(
#                D_real_Body=torch.sigmoid(D_real_Body).mean().item(),
#                D_fake_Body=torch.sigmoid(D_fake_Body).mean().item(),
#                D_real_GTV=torch.sigmoid(D_real_GTV).mean().item(),
#                D_fake_GTV=torch.sigmoid(D_fake_GTV).mean().item(),                
#            )


def main():
    disc_Body = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    disc_Body=disc_Body.double()
    disc_GTV = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    disc_GTV =disc_GTV.double()
    gen = Generator(in_channels=config.CHANNELS_IMG, features=32).to(config.DEVICE)
    gen=gen.double()
    opt_disc_Body = optim.Adam(disc_Body.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_disc_GTV = optim.Adam(disc_GTV.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss().to(config.DEVICE)
    L2_LOSS = nn.MSELoss().to(config.DEVICE)
    SSIM_LOSS = pytorch_ssim.SSIM()
    
    folder_imgs="evaluationTrials/l2/images/"
    folder_matfiles="evaluationTrials/l2/matfiles/"
    os.mkdir(folder_imgs+config.EXP_NAME)
    os.mkdir(folder_matfiles+config.EXP_NAME)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )


    g_cpu_train1 = torch.Generator()
    g_cpu_train2 = torch.Generator()
    g_cpu_train3 = torch.Generator()
    g_cpu_train4 = torch.Generator()
    g_cpu_train5 = torch.Generator()    

    g_cpu_train1.set_state(g_cpu_train5.get_state())
    g_cpu_train2.set_state(g_cpu_train5.get_state())
    g_cpu_train3.set_state(g_cpu_train5.get_state())
    g_cpu_train4.set_state(g_cpu_train5.get_state())
    
    train_input_dataset = MapDataset_input(input_root_dir=config.TRAIN_INPUT_DIR)
    train_input_dataset.input_list_files = sorted(train_input_dataset.input_list_files)#######
#    train_input_dataset.input_list_files = train_input_dataset.input_list_files[0:8]#######
    train_input_loader = DataLoader(train_input_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True,generator = g_cpu_train1,num_workers=config.NUM_WORKERS)
    train_organMap_dataset = MapDataset_organMap(organMap_root_dir=config.TRAIN_ORGANMAP_DIR)
    train_organMap_dataset.organMap_list_files = sorted(train_organMap_dataset.organMap_list_files)######
#    train_organMap_dataset.organMap_list_files = train_organMap_dataset.organMap_list_files[0:8]######    
    train_organMap_loader = DataLoader(train_organMap_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True,generator = g_cpu_train2,num_workers=config.NUM_WORKERS)
    train_mask_dataset = MapDataset_mask(mask_root_dir=config.TRAIN_MASK_DIR)
    train_mask_dataset.mask_list_files = sorted(train_mask_dataset.mask_list_files)######   
#    train_mask_dataset.mask_list_files = train_mask_dataset.mask_list_files[0:8]######      
    train_mask_loader = DataLoader(train_mask_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True,generator = g_cpu_train3,num_workers=config.NUM_WORKERS) 
    train_bodyMask_dataset = MapDataset_bodyMask(bodyMask_root_dir=config.TRAIN_BODYMASK_DIR)
    train_bodyMask_dataset.bodyMask_list_files = sorted(train_bodyMask_dataset.bodyMask_list_files)######   
#    train_bodyMask_dataset.bodyMask_list_files = train_bodyMask_dataset.bodyMask_list_files[0:8]######      
    train_bodyMask_loader = DataLoader(train_bodyMask_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True,generator = g_cpu_train4,num_workers=config.NUM_WORKERS)                              
    train_target_dataset = MapDataset_target(target_root_dir=config.TRAIN_TARGET_DIR)
    train_target_dataset.target_list_files = sorted(train_target_dataset.target_list_files)######
#    train_target_dataset.target_list_files = train_target_dataset.target_list_files[0:8]######
    train_target_loader = DataLoader(train_target_dataset, batch_size=config.BATCH_SIZE,
                                     shuffle=True,generator = g_cpu_train5,num_workers=config.NUM_WORKERS)
    
 



    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler_Body = torch.cuda.amp.GradScaler()
    d_scaler_GTV = torch.cuda.amp.GradScaler()
    
    g_cpu = torch.Generator()
    g_cpu_other1 = torch.Generator()
    g_cpu_other2 = torch.Generator()
    g_cpu.set_state(g_cpu_other2.get_state())
    g_cpu_other1.set_state(g_cpu_other2.get_state())
    
    
#    val_input_dataset = MapDataset_input(input_root_dir=config.VAL_INPUT_DIR)
#    val_input_loader = DataLoader(val_input_dataset, batch_size=config.BATCH_SIZE,
#                                    shuffle=False,num_workers=config.NUM_WORKERS)
    val_organMap_dataset = MapDataset_organMap(organMap_root_dir=config.VAL_ORGANMAP_DIR)
    val_organMap_dataset.organMap_list_files = sorted(val_organMap_dataset.organMap_list_files)
    val_organMap_loader = DataLoader(val_organMap_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True, generator=g_cpu, num_workers=config.NUM_WORKERS)
#    val_mask_dataset = MapDataset_mask(mask_root_dir=config.VAL_MASK_DIR)
#    val_mask_loader = DataLoader(val_mask_dataset, batch_size=config.BATCH_SIZE,
#                                    shuffle=False,num_workers=config.NUM_WORKERS)
    val_target_dataset = MapDataset_target(target_root_dir=config.VAL_TARGET_DIR)
    val_target_dataset.target_list_files = sorted(val_target_dataset.target_list_files)
    val_target_loader = DataLoader(val_target_dataset, batch_size=config.BATCH_SIZE,
                                     shuffle=True, generator=g_cpu_other1, num_workers=config.NUM_WORKERS)
    val_bodyMask_dataset = MapDataset_bodyMask(bodyMask_root_dir=config.VAL_BODYMASK_DIR) 
    val_bodyMask_dataset.bodyMask_list_files = sorted(val_bodyMask_dataset.bodyMask_list_files)         
    val_bodyMask_loader = DataLoader(val_bodyMask_dataset, batch_size=config.BATCH_SIZE,
                                    shuffle=True,generator = g_cpu_other2,num_workers=config.NUM_WORKERS)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_Body,disc_GTV, gen, train_input_loader, train_organMap_loader, train_mask_loader, train_bodyMask_loader,
            train_target_loader,opt_disc_Body, opt_disc_GTV, opt_gen, L1_LOSS, L2_LOSS, SSIM_LOSS, BCE, g_scaler, 
            d_scaler_Body, d_scaler_GTV,epoch,
        )
        writer.flush()

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc_Body, opt_disc_Body, filename=config.CHECKPOINT_DISC_BODY)
            save_checkpoint(disc_GTV, opt_disc_GTV, filename=config.CHECKPOINT_DISC_GTV)

            save_some_examples(gen, val_organMap_loader, val_target_loader, val_bodyMask_loader,
                               epoch,writer, folder_imgs="evaluationTrials/l2/images/",
                               folder_matfiles="evaluationTrials/l2/matfiles/")


if __name__ == "__main__":
    main()


# In[ ]:


writer.close()


# In[ ]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')
#get_ipython().run_line_magic('tensorboard', '--logdir=runs')

