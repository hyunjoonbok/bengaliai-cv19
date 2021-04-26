from albumentations.augmentations.transforms import Normalize, Resize
from albumentations.core.composition import set_always_apply
import numpy as np
import os
import gc
from numpy.core.fromnumeric import mean
import pandas as pd
import joblib 
from PIL import Image

import torch
from torch.utils.data import Dataset

# Image Transformation
import albumentations
from albumentations.pytorch import ToTensor

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, 
    ElasticTransform, ChannelShuffle,RGBShift, Rotate, Cutout
)

# YOUR PATH
path = f'/mnt/c/Users/bokhy/Desktop/Python/github/kaggle/bengaliai-cv19/input/'
HEIGHT = 137
WIDTH = 236
# Augmentation
train_augmentation = Compose([
        Resize(HEIGHT,WIDTH, always_apply= True),
        Normalize(mean, std, always_apply = True),
        Rotate(),
        Flip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        ToTensor()
    ])

# No Augmentation to Validset
valid_augmentation = Compose([
        ToTensor()
    ])

class BengaliDataset:
    
    # I define the csv file and the height and width of the image
    # mean and std is needed for normalization in image problem
    def __init__(self, fold, img_height, img_width, transform):
        # To make sure having ordered index with the 'fold'
        df = pd.read_csv(os.path.join(path, 'df_folds.csv'))
        # Get the targets 
        df = df[["image_id", "grapheme_root","vowel_diacritic","consonant_diacritic", "fold"]]
        
        # Get certain fold s
        df = df[df.folds.isin(fold)].reset_index(drop = True)
        
        # Convert every column to numpy array
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.image_id.values
        self.consonant_diacritic = df.consonant_diacritic.values
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.transform = transform
    
    # The lenght of the whole data --> In our case, the lengh of the train CSV file
    def __len__(self):
        return len(self.image_ids)
    
    # How are we going to process the image and get the final output image --> 'Image Precessing Steps' above
    def __getitem__(self, item):
        img_id = self.image_ids[item]
        img = joblib.load(os.path.join(path, f'train_images/{img_id}.pkl'))
        # Image is a vector --> Reshape it to 2d array
        img = img.reshape(self.img_height, self.img_width).astype(np.uint8)
        img = 255 - img
        # Convert Grey image to RGB for easier transfer learning 
        # (Most of Pretrained Image Models are in Color)
        img = Image.fromarray(img).convert("RGB")
        img = img[:, : , np.newaxis] 
        
        
        # Fianl output should be able to be read in Pytorch --> (Batch, Channel, Height, Width)
        return ('image': torch.tensor(img, dtype = torch.float).permute(2,0,1), 
               'grapheme_root': torch.tensor(self.grapheme_root[item], dtype = torch.long),
               'vowel_diacritic': torch.tensor(self.vowel_diacritic[item], dtype = torch.long),
               'consonant_diacritic': torch.tensor(self.consonant_diacritic[item], dtype = torch.long))