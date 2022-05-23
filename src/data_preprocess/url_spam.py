# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:50:58 2022

@author: YingFu
"""
import os
import pandas as pd

if __name__ == "__main__":
    import data_util # 单独运行本脚本用
else:
    from . import data_util  # 当作一个module被别人 import 时用 


def load(data_dir="../../data",drop_use_less=True,take_sample=True,trainTestSeedLst=[1,2,3,4,5]):    
    file_name = os.path.join(data_dir, 'url_spam_classification/url_spam_classification.csv') 
    df = pd.read_csv(file_name)
    # print(df.info())
    # RangeIndex: 148303 entries, 0 to 148302
    # Data columns (total 2 columns):
    #  #   Column   Non-Null Count   Dtype 
    # ---  ------   --------------   ----- 
    #  0   url      148303 non-null  object
    #  1   is_spam  148303 non-null  bool  
    # dtypes: bool(1), object(1)
    
    # extract features based on url
        
    df['len_url'] = df['url'].apply(lambda x : len(x))  
    df['contains_subscribe'] = df['url'].apply(lambda x : 1 if "subscribe" in x else 0)
    df['contains_hash'] = df['url'].apply(lambda x : 1 if "#" in x else 0)
    df['num_digits'] = df['url'].apply(lambda x : len("".join(_ for _ in x if _.isdigit())) )
    df['non_https'] = df['url'].apply(lambda x : 1 if "https" in x else 0)
    df['num_words'] = df['url'].apply(lambda x : len(x.split("/")))
    df['contains_?'] = df['url'].apply(lambda x : 1 if "?" in x else 0)
    df['contains_www'] = df['url'].apply(lambda x : 1 if "www" in x else 0)

    y_col = 'spam'
    df[y_col] = df['is_spam'].apply(lambda x: 1 if x == True else 0)  
    df = df.drop('is_spam', 1)
    df = df.drop('url', 1)
    
    useless_cols_dict = data_util.find_useless_colum(df)
    df = data_util.drop_useless(df, useless_cols_dict)
    df = df.reset_index(drop=True)
    print(df.info())
    # Data columns (total 9 columns):
    #  #   Column              Non-Null Count   Dtype
    # ---  ------              --------------   -----
    #  0   len_url             148303 non-null  int64
    #  1   contains_subscribe  148303 non-null  int64
    #  2   contains_hash       148303 non-null  int64
    #  3   num_digits          148303 non-null  int64
    #  4   non_https           148303 non-null  int64
    #  5   num_words           148303 non-null  int64
    #  6   contains_?          148303 non-null  int64
    #  7   contains_www        148303 non-null  int64
    #  8   spam                148303 non-null  int64
    
    # take a small sample
    if take_sample:
        df_positive = df.query('spam == 1 ').sample(n=8000,random_state=1)
        df_negative = df.query('spam == 0 ').sample(n=8000,random_state=1)
        df = pd.concat([df_positive,df_negative]).reset_index(drop=True)
    
    ds = data_util.DataSetPreprocess("url_spam", df)
    ds_list = ds.preprocess(trainTestSeedLst)
    
    return ds_list


if __name__ == "__main__":
    ds_list_url_spam = load()



