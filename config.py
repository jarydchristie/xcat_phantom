#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_INPUT_DIR = "data/finalData_crop/train/simulatedImage"
TRAIN_TARGET_DIR = "data/finalData_crop/train/realImage"
TRAIN_ORGANMAP_DIR = "data/finalData_crop/train/organMap"
TRAIN_MASK_DIR = "data/finalData_crop/train/mask"
VAL_INPUT_DIR = "data/finalData_crop/val/simulatedImage"
VAL_TARGET_DIR = "data/finalData_crop/val/realImage"
VAL_ORGANMAP_DIR = "data/finalData_crop/val/organMap"
VAL_MASK_DIR = "data/finalData_crop/val/mask"
LEARNING_RATE = 2e-4#0.001
BATCH_SIZE = 2
NUM_WORKERS = 4
#IMAGE_SIZE = 512
CHANNELS_IMG = 1
L1_LAMBDA = 2
SSIM_MU = 5
SSIM_V = 1000
#LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

#both_transform = A.Compose(
#    [A.Resize(width=512, height=512),], additional_targets={"image0": "image"},
#)

#transform_only_input = A.Compose(
#    [
#        A.HorizontalFlip(p=0.5),
#        A.ColorJitter(p=0.2),
#        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#        ToTensorV2(),
#    ]
#)

#transform_only_mask = A.Compose(
#    [
#        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#        ToTensorV2(),
#    ]
#)

