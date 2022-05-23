# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:12:54 2022

@author: YingFu
"""

import pandas as pd
import os
if __name__ == "__main__":
    import data_util # 单独运行本脚本用
else:
    from . import data_util  # 当作一个module被别人 import 时用 

def load(data_dir='../../data', drop_use_less=True,trainTestSeedLst=[1,2,3,4,5]): 
    file_name = os.path.join(data_dir, 'mushrooms/mushrooms.csv')
    df = pd.read_csv(file_name)
    y = df.iloc[:,-1]
    new_y = y.replace({'e':1}, regex=True)
    new_y = new_y.replace({'p':0}, regex=True)
    df.iloc[:,-1] = new_y
    for col in df:        
        df[col] = df[col].astype('category')        
    df = df.reset_index(drop=True)
    
    print(df.iloc[:,-1].value_counts(sort=False,dropna=False))
    # 0    3916
    # 1    4208
    
    # print(df.info())
    # Data columns (total 23 columns):
    #  #   Column                    Non-Null Count  Dtype   
    # ---  ------                    --------------  -----   
    #  0   cap-shape                 8124 non-null   category
    #  1   cap-surface               8124 non-null   category
    #  2   cap-color                 8124 non-null   category
    #  3   bruises                   8124 non-null   category
    #  4   odor                      8124 non-null   category
    #  5   gill-attachment           8124 non-null   category
    #  6   gill-spacing              8124 non-null   category
    #  7   gill-size                 8124 non-null   category
    #  8   gill-color                8124 non-null   category
    #  9   stalk-shape               8124 non-null   category
    #  10  stalk-root                8124 non-null   category
    #  11  stalk-surface-above-ring  8124 non-null   category
    #  12  stalk-surface-below-ring  8124 non-null   category
    #  13  stalk-color-above-ring    8124 non-null   category
    #  14  stalk-color-below-ring    8124 non-null   category
    #  15  veil-type                 8124 non-null   category
    #  16  veil-color                8124 non-null   category
    #  17  ring-number               8124 non-null   category
    #  18  ring-type                 8124 non-null   category
    #  19  spore-print-color         8124 non-null   category
    #  20  population                8124 non-null   category
    #  21  habitat                   8124 non-null   category
    #  22  class                     8124 non-null   category
    
    mushroomdataset = data_util.DataSetPreprocess("mushroom", df)
    print(f"{mushroomdataset.npratio = }")
    ds_list = mushroomdataset.preprocess(trainTestSeedLst)
    
    return ds_list


if __name__ == "__main__":
    ds_list_mushroom = load(trainTestSeedLst=[4,5])