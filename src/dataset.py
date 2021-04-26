import numpy as np
import os
import gc
import pandas as pd
import joblib 

import torch
from torch.utils.data import Dataset

# Image Transformation
import albumentations as A
from albumentations.pytorch import ToTensor

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, 
    ElasticTransform, ChannelShuffle,RGBShift, Rotate, Cutout
)

# YOUR PATH
path = f'/mnt/c/Users/bokhy/Desktop/Python/github/kaggle/bengaliai-cv19/input/'

class BengaliDataset(Dataset):
    
    # I define the csv file and the height and width of the image
    # mean and std is needed for normalization in image problem
    def __init__(self, csv, img_height, img_width, mean, std):
        # To make sure having ordered index with the 'fold'
        self.csv = csv.reset_index()
        self.img_ids = csv['image_id'].values
        self.img_height = img_height
        self.img_width = img_width
    
    # The lenght of the whole data --> In our case, the lengh of the train CSV file
    def __len__(self):
        return len(self.csv)
    
    # How are we going to process the image and get the final output image --> 'Image Precessing Steps' above
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = joblib.load(os.path.join(path, f'train_images/{img_id}.pkl'))
        img = img.reshape(self.img_height, self.img_width).astype(np.uint8)
        img = 255 - img
        img = img[:, : , np.newaxis]
        
        label_1 = self.csv.iloc[index].grapheme_root
        label_2 = self.csv.iloc[index].vowel_diacritic
        label_3 = self.csv.iloc[index].consonant_diacritic
        
        # Fianl output should be able to be read in Pytorch --> (Batch, Channel, Height, Width)
        return (torch.tensor(img, dtype = torch.float).permute(2,0,1), 
               torch.tensor(label_1, dtype = torch.long),
               torch.tensor(label_2, dtype = torch.long),
               torch.tensor(label_3, dtype = torch.long))