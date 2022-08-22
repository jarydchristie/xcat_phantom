#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from datetime import datetime
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_INPUT_DIR = "data/finalData_crop4/train/simulatedImage"
TRAIN_TARGET_DIR = "data/finalData_crop4/train/realImage"
TRAIN_ORGANMAP_DIR = "data/finalData_crop4/train/organMap"
TRAIN_MASK_DIR = "data/finalData_crop4/train/mask"
TRAIN_BODYMASK_DIR = "data/finalData_crop4/train/bodyMask"
VAL_INPUT_DIR = "data/finalData_crop4/val/simulatedImage"
VAL_TARGET_DIR = "data/finalData_crop4/val/realImage"
VAL_ORGANMAP_DIR = "data/finalData_crop4/val/organMap"
VAL_MASK_DIR = "data/finalData_crop4/val/mask"
VAL_BODYMASK_DIR = "data/finalData_crop4/val/bodyMask"

EXP_NAME = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
EXP_NAME = EXP_NAME.replace(":","")
EXP_NAME = EXP_NAME.replace(".","")
EXP_NAME = EXP_NAME.replace(" ","")

LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_WORKERS = 4
#IMAGE_SIZE = 512
CHANNELS_IMG = 1
L1_LAMBDA = 2
SSIM_MU = 5
SSIM_V = 100
#LAMBDA_GP = 10
NUM_EPOCHS = 4000
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC_BODY = EXP_NAME+"discbody.pth.tar"
CHECKPOINT_DISC_GTV = EXP_NAME+"discgtv.pth.tar"
CHECKPOINT_GEN = EXP_NAME+"gen.pth.tar"

