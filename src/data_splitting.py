# Funciones para dividir los datos

import numpy as np

def train_val_split(df, train_ratio=0.8, random_state=42):
    np.random.seed(random_state) # set a seed for reproducibility
    shuffled_indexes = np.random.permutation(len(df))

    # set train and val size

    train_size = int(len(df) * train_ratio)

    # set indexes
    train_idx = shuffled_indexes[:train_size]
    val_idx = shuffled_indexes[train_size:]

    # create subsets and reset indexes to start from 0 and drop old ones

    train_set = df.iloc[train_idx].copy().reset_index(drop=True)
    val_set = df.iloc[val_idx].copy().reset_index(drop=True)

    return train_set, val_set
   
def cross_val():
    pass