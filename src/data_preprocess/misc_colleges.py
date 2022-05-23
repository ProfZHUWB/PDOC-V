# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:49:52 2021

@author: iwenc
"""

import os
import pandas as pd

if __name__ == "__main__":
    import data_util # 单独运行本脚本用
else:
    from . import data_util  # 当作一个module被别人 import 时用 


def load(data_dir="../../data", drop_use_less=True,trainTestSeedLst=[1,2,3,4,5]):
    file_name = os.path.join(data_dir,'misc/Colleges/Colleges.txt')
    df = pd.read_csv(file_name, sep='\t', encoding='latin1', na_values=['','PrivacySuppressed'], index_col=0)

    # column                        dtype       na  physcial_type  desc
    # UNITID                        int64           id             
    # School Name                   category
    # City                          category
    # State                         category
    # ZIP                           category
    # School Webpage                category    ''
    # Latitude                      float64     ''  GPS position
    # Longitude                     float64     ''  GPS position
    # Admission Rate                float64	    ''  rate[0,1.0] 
    # SAT Verbal Midrange           float64     ''  uint16
    # SAT Math Midrange             float64     ''  uint16
    # SAT Writing Midrange          float64     ''  uint16
    # ACT Combined Midrange         float64     ''  uint8
    # ACT English Midrange          float64     ''  uint8
    # ACT Math Midrange             float64     ''  uint8
    # ACT Writing Midrange          float64     ''  uint8
    # SAT Total Average             float64     ''  uint16
    # Undergrad Size                float64     ''  uint32
    # Percent White                 float64     ''  rate[0,1.0] 
    # Percent Black                 float64     ''  rate[0,1.0] 
    # Percent Hispanic              float64     ''  rate[0,1.0] 
    # Percent Asian                 float64     ''  rate[0,1.0] 
    # Percent Part Time             float64     ''  rate[0,1.0] 
    # Average Cost Academic Year    float64     ''  money
    # Average Cost Program Year     float64     ''  money
    # Tuition (Instate)             float64     ''  money
    # Tuition (Out of state)        float64     ''  money
    # Spend per student             float64     ''  money
    # Faculty Salary                float64     ''  money
    # Percent Part Time Faculty     float64     ''  rate[0,1.0] 
    # Percent Pell Grant            float64     ''  rate[0,1.0] 
    # Completion Rate               float64     ''  rate[0,1.0] 
    # Average Age of Entry          float64     ''  rate[0,1.0] 
    # Percent Married               float64     ''  rate[0,1.0] 
    # Percent Veteran               float64     ''  rate[0,1.0] 
    # Predominant Degree            category    'None'
    # Highest Degree                category
    # Ownership                     category
    # Region                        category
    # Gender                        category
    # Carnegie Basic Classification category    ''
    # Carnegie Undergraduate        category    ''
    # Carnegie Size                 category    ''
    # Religious Affiliation         category    ''
    # Percent Female                float64     '', 'PrivacySuppressed' rate[0,1.0] 
    # agege24                       float64     '', 'PrivacySuppressed' rate[0,1.0] 
    # faminc                        float64     '', 'PrivacySuppressed' rate[0,1.0] 
    # Mean Earnings 6 years         float64     '', 'PrivacySuppressed' rate[0,1.0] 
    # Median Earnings 6 years       float64     '', 'PrivacySuppressed' rate[0,1.0] 
    # Mean Earnings 10 years        float64     '', 'PrivacySuppressed' rate[0,1.0] 
    # Median Earnings 10 years      float64     '', 'PrivacySuppressed' rate[0,1.0] 

    cat_cols = ['School Name', 'City', 'State', 'ZIP', 'School Webpage',
                'Predominant Degree', 'Highest Degree', 'Ownership', 'Region',
                'Gender', 'Carnegie Basic Classification',
                'Carnegie Undergraduate', 'Carnegie Size',
                'Religious Affiliation']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # find columns whose data type need to be explicitly converetd
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f'{col}:{df[col].dtype}')

    # convert into binary classification, 1 if `Mean Earnings 6 years' > 30000
    y_col = 'MeanEarning6Year>=30000'
    df['MeanEarning6Year>=30000'] = df['Mean Earnings 6 years'].apply(lambda x: 1 if x>=30000 else 0)  
    
    if drop_use_less:
        # drop id column, unique value for each record
        df = df.drop(['UNITID'], axis=1)        
        
        # earning over 6 years derives the target variable, they should be deleted
        # earning over 10 years leaks information when predicting earnings over 6 years, they should be deleted
        df = df.drop(['Mean Earnings 6 years', 'Median Earnings 6 years',
                  'Mean Earnings 10 years', 'Median Earnings 10 years'], axis=1)
        
        # drop empty columns: ['Average Age of Entry', 'Percent Married', 'Percent Veteran']
        # drop columns with more than 50% missing: ['Admission Rate', 'SAT Verbal Midrange', 'SAT Math Midrange', 'SAT Writing Midrange', 'ACT Combined Midrange', 'ACT English Midrange', 'ACT Math Midrange', 'ACT Writing Midrange', 'SAT Total Average', 'Average Cost Program Year', 'Completion Rate', 'Carnegie Undergraduate', 'Carnegie Size', 'Religious Affiliation']
        # drop columns with average sample per category < 2: ['School Name', 'ZIP', 'School Webpage']
        # drop columns with a single category more than 90% samples: ['Gender']
        useless_cols_dict = data_util.find_useless_colum(df)
        df = data_util.drop_useless(df, useless_cols_dict)
    
    df = df.reset_index(drop=True)
    # print(df[y_col].value_counts(sort=False,dropna=False))
    # 0    5352
    # 1    2452
    # print(df.info())
    # RangeIndex: 7804 entries, 0 to 7803
    # Data columns (total 26 columns):
    #  #   Column                         Non-Null Count  Dtype   
    # ---  ------                         --------------  -----   
    #  0   City                           7804 non-null   category
    #  1   State                          7804 non-null   category
    #  2   Latitude                       7019 non-null   float64 
    #  3   Longitude                      7019 non-null   float64 
    #  4   Undergrad Size                 7090 non-null   float64 
    #  5   Percent White                  7090 non-null   float64 
    #  6   Percent Black                  7090 non-null   float64 
    #  7   Percent Hispanic               7090 non-null   float64 
    #  8   Percent Asian                  7090 non-null   float64 
    #  9   Percent Part Time              7072 non-null   float64 
    #  10  Average Cost Academic Year     4137 non-null   float64 
    #  11  Tuition (Instate)              4415 non-null   float64 
    #  12  Tuition (Out of state)         4196 non-null   float64 
    #  13  Spend per student              7362 non-null   float64 
    #  14  Faculty Salary                 4654 non-null   float64 
    #  15  Percent Part Time Faculty      4127 non-null   float64 
    #  16  Percent Pell Grant             7063 non-null   float64 
    #  17  Predominant Degree             7804 non-null   category
    #  18  Highest Degree                 7804 non-null   category
    #  19  Ownership                      7804 non-null   category
    #  20  Region                         7804 non-null   category
    #  21  Carnegie Basic Classification  4355 non-null   category
    #  22  Percent Female                 5713 non-null   float64 
    #  23  agege24                        5713 non-null   float64 
    #  24  faminc                         5713 non-null   float64 
    #  25  MeanEarning6Year>=30000        7804 non-null   int64   
    
    ds = data_util.DataSetPreprocess("colleges", df)
    ds_list = ds.preprocess(trainTestSeedLst)
    
    return ds_list
    

if __name__ == "__main__":
    ds_list_colleges = load()


