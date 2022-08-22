#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import config
import os
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
#from torchvision.utils import save_image

class MapDataset_input(Dataset):
    def __init__(self, input_root_dir):
        self.input_root_dir = input_root_dir
        self.input_list_files = os.listdir(self.input_root_dir)      

    def __len__(self):
        return len(self.input_list_files)

    def __getitem__(self, index):
        input_img_file = self.input_list_files[index]
        input_img_path = os.path.join(self.input_root_dir, input_img_file)       
        
        input_Body_mat = sio.loadmat(input_img_path)
        
        input_Body_mat_data = input_Body_mat['simulatedImageBody_slice']
        
        input_Body_image = np.fliplr(np.rot90(np.transpose(input_Body_mat_data),3))


        return input_Body_image

class MapDataset_target(Dataset):
    def __init__(self, target_root_dir):
        self.target_root_dir = target_root_dir
        self.target_list_files = os.listdir(self.target_root_dir)        

    def __len__(self):
        return len(self.target_list_files)

    def __getitem__(self, index):
        
        target_img_file = self.target_list_files[index]
        target_img_path = os.path.join(self.target_root_dir, target_img_file)        
        
        target_Body_mat = sio.loadmat(target_img_path)
        
        target_Body_mat_data = target_Body_mat['realImageBody_slice']
        
        target_Body_image = np.fliplr(np.rot90(np.transpose(target_Body_mat_data),3))
        

        return target_Body_image
    
class MapDataset_mask(Dataset):
    def __init__(self, mask_root_dir):
        self.mask_root_dir = mask_root_dir
        self.mask_list_files = os.listdir(self.mask_root_dir)        

    def __len__(self):
        return len(self.mask_list_files)

    def __getitem__(self, index):
        
        mask_img_file = self.mask_list_files[index]
        mask_img_path = os.path.join(self.mask_root_dir, mask_img_file)        
        
        mask_mat = sio.loadmat(mask_img_path)
        
        mask_mat_data = mask_mat['tumourSeg_Slice']
        
        mask_image = np.fliplr(np.rot90(np.transpose(mask_mat_data),3))
        
        return mask_image

class MapDataset_bodyMask(Dataset):
    def __init__(self, bodyMask_root_dir):
        self.bodyMask_root_dir = bodyMask_root_dir
        self.bodyMask_list_files = os.listdir(self.bodyMask_root_dir)        

    def __len__(self):
        return len(self.bodyMask_list_files)

    def __getitem__(self, index):
        
        bodyMask_img_file = self.bodyMask_list_files[index]
        bodyMask_img_path = os.path.join(self.bodyMask_root_dir, bodyMask_img_file)        
        
        bodyMask_mat = sio.loadmat(bodyMask_img_path)
        
        bodyMask_mat_data = bodyMask_mat['bodyMask']
        
        bodyMask_image = np.fliplr(np.rot90(np.transpose(bodyMask_mat_data),3))
        
        return bodyMask_image 

class MapDataset_organMap(Dataset):
    def __init__(self, organMap_root_dir):
        self.organMap_root_dir = organMap_root_dir
        self.organMap_list_files = os.listdir(self.organMap_root_dir)        

    def __len__(self):
        return len(self.organMap_list_files)

    def __getitem__(self, index):
        
        organMap_img_file = self.organMap_list_files[index]
        organMap_img_path = os.path.join(self.organMap_root_dir, organMap_img_file)        
        
        organMap_Body_mat = sio.loadmat(organMap_img_path)
        
        organMap_Body_mat_data = organMap_Body_mat['organMapBody_slice']
        
        organMap_Body_image = np.fliplr(np.rot90(np.transpose(organMap_Body_mat_data),3))
       

        return organMap_Body_image
#if __name__ == "__main__":
#    dataset_input = MapDataset_input("data/dataBody/train/simulatedImage")
#    dataset_target = MapDataset_target("data/dataBody/train/realImage")
#    loader_input = DataLoader(dataset_input, batch_size=1)
#    loader_target = DataLoader(dataset_target, batch_size=1)
#    for x in loader_input:
        #print(x.shape)
#        plt.imsave('input.png', x[0,:,:], cmap='gray')
        #save_image(x, "x.png")
        #save_image(y, "y.png")
        #import sys

        #sys.exit()
#    for x in loader_target:
        #print(x.shape)
#        plt.imsave('target.png', x[0,:,:], cmap='gray')
        #save_image(x, "x.png")
        #save_image(y, "y.png")
        #import sys

        #sys.exit()        

