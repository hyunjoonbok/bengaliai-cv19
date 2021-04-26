import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    # Linux Path (WSL)
    path = f'/mnt/c/Users/bokhy/Desktop/Python/github/kaggle/bengaliai-cv19/input/'
    df_train = pd.read_csv(os.path.join(path, 'train.csv'))
    
    # Create IDs in train set 
    # # We can easily extract numbers from the 'image_id' column
    df_train['id'] = df_train['image_id'].apply(lambda x: int(x.split('_')[1]))
    
    # Get values of first column (just 'id')
    X = df_train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:, 0]

    # get values from second column ('grapheme_root', 'vowel_diacritic', 'consonant_diacritic')
    y = df_train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:, 1:]

    mskf = MultilabelStratifiedKFold(n_splits=6, shuffle=True, random_state=623)
    
    df_train['fold'] = 'int' #placeholder

    for i, (trn_idx, vld_idx) in enumerate(mskf.split(X, y)):
        print("TRAIN:", trn_idx, "TEST:", vld_idx)
        df_train.loc[vld_idx, 'fold'] = i
        
    df_train.to_csv(os.path.join(path, 'df_folds.csv'), index = False)
