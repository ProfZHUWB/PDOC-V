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

def load(data_dir='../../data', drop_use_less=True,trainTestSeedLst = [1,2,3,4,5]): 
    
    file_name = os.path.join(data_dir, 'employee_salaries/rows.csv') 
    df = pd.read_csv(file_name)
    #  #   Column                   Non-Null Count  Dtype         
    # ---  ------                   --------------  -----         
    #  0   Full Name                9228 non-null   category      
    #  1   Gender                   9211 non-null   category      
    #  2   Current Annual Salary    9228 non-null   float64       
    #  3   2016 Gross Pay Received  9128 non-null   float64       
    #  4   2016 Overtime Pay        6311 non-null   float64       
    #  5   Department               9228 non-null   category      
    #  6   Department Name          9228 non-null   category      
    #  7   Division                 9228 non-null   category      
    #  8   Assignment Category      9228 non-null   category      
    #  9   Employee Position Title  9228 non-null   category      
    #  10  Underfilled Job Title    1093 non-null   category      
    #  11  Date First Hired         9228 non-null   datetime64[ns] 

    cat_cols = ['Full Name', 'Gender', 'Department', 'Department Name', 
                'Division', 'Assignment Category', 'Employee Position Title', 
                'Underfilled Job Title']
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
    
    time_cols = ['Date First Hired']
    for col in time_cols:
         df[col] =  pd.to_datetime(df[col])

    # print(df.info())
    
    y_col = 'Current Annual Salary>=100000'
    df[y_col] = df['Current Annual Salary'].apply(lambda x: 1 if x>=100000 else 0)  
    df = df.drop('Current Annual Salary', 1)

    # Department 是 Department Name的缩写，两者一一对应，删掉其中一列
    # print(f"{util.is_one_to_one(df, 'Department', 'Department Name')=}")  # Output one-to-one; or None
    df = df.drop('Department', 1)

    if drop_use_less:
        useless_cols_dict = data_util.find_useless_colum(df)
        df = data_util.drop_useless(df, useless_cols_dict)
        
    # 有一列是日期，找到最大日期的那个人，然后用那个人的日期减去其他人，作为工作时长基准，并且删掉日期列
    most_recent_date = df['Date First Hired'].max()
    print(f"{most_recent_date = }")
    df['working days'] = (most_recent_date - df['Date First Hired']).dt.days  
    df = df.drop('Date First Hired', 1)
    # print(df.info())
    #      #   Column                         Non-Null Count  Dtype   
    # ---  ------                         --------------  -----   
    #  0   Gender                         9211 non-null   category
    #  1   2016 Gross Pay Received        9128 non-null   float64 
    #  2   2016 Overtime Pay              6311 non-null   float64 
    #  3   Department Name                9228 non-null   category
    #  4   Division                       9228 non-null   category
    #  5   Employee Position Title        9228 non-null   category
    #  6   Current Annual Salary>=100000  9228 non-null   int64   
    #  7   working days                   9228 non-null   int64   
    obj = df['Current Annual Salary>=100000']
    df = df.drop('Current Annual Salary>=100000', 1)
    df['y'] = obj
    
    df = df.reset_index(drop=True)
    print(df['y'].value_counts(sort=False,dropna=False))
    # 0    7745
    # 1    1483
    # print(df.info())
    # RangeIndex: 9228 entries, 0 to 9227
    # Data columns (total 8 columns):
    #  #   Column                   Non-Null Count  Dtype   
    # ---  ------                   --------------  -----   
    #  0   Gender                   9211 non-null   category
    #  1   2016 Gross Pay Received  9128 non-null   float64 
    #  2   2016 Overtime Pay        6311 non-null   float64 
    #  3   Department Name          9228 non-null   category
    #  4   Division                 9228 non-null   category
    #  5   Employee Position Title  9228 non-null   category
    #  6   working days             9228 non-null   int64   
    #  7   y                        9228 non-null   int64   
    

    ds = data_util.DataSetPreprocess("employee_salaries", df)
    ds_list = ds.preprocess(trainTestSeedLst)
    
    return ds_list
    

if __name__ == "__main__":
    ds_list_employee_salaries = load()
    