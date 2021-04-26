import os
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import joblib 
from tqdm import tqdm

if __name__ == "__main__":
    # Linux Path (WSL)
    path = f'/mnt/c/Users/bokhy/Desktop/Python/github/kaggle/bengaliai-cv19/input/'
    
    files_train = [f'/mnt/c/Users/bokhy/Desktop/Python/github/kaggle/bengaliai-cv19/input/train_image_data_{imid}.parquet' for imid in range(4)]

    for fname in files_train:
        # Read all four train_image_data parquet files
        df_train = pd.read_parquet(fname, engine = 'fastparquet')
        
        # Each row with image_id indicates each image (in pixels)
        # So we will separate each of the row and iterate it for more efficient data loading
        
        # To do so, we change the pandas dataset to numpy array to speed things up
        img_ids = df_train['image_id'].values
        img_array = df_train.iloc[:, 1:].values
        
        # Save the ids and arrays into Python pickle file
        for idx in tqdm(range(len(df_train))):
            img_id = img_ids[idx]
            img = img_array[idx]
            joblib.dump(img, f'/mnt/c/Users/bokhy/Desktop/Python/github/kaggle/bengaliai-cv19/input/train_images/{img_id}.pkl')