# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:11:28 2021

@author: iwenc
"""

import numpy as np
import pandas as pd

def find_useless_colum(df, max_missing_ratio=0.5, min_rows_per_value=2, max_ratio_per_cat=0.9, verbose=False):
    '''
    Identify useless columns from a DataFrame. 
    Columns are divided into three types:
        num_float: contains numeric values only and at least a number that is not integer
        num_int:   contains numeric values and all values are integers (3.0 is treated as an integer)
        cat_like:  none numeric values are considered like categorical            
    
    If column is considered useless and classified into the following types:
        empty:              if the column contains no value
        singel-valued:      if the column contains only one value
        id-like:            A num_int or cat_like column contains a unique
                            value for each sample. It is okay for a num_float
                            column to contain a unqiue value for each sample
        too-many-missing:   if a column contains too many missing values --
                            exceeding total number of samples * max_missing_ratio
        too-small-cat:      if average samples per category are too few in a
                            cat_like column -- less than min_rows_per_value
        too-large-cat:      if a single category in a cat-like column contains
                            too many samples -- exceeding total number of
                            samples * max_ratio_per_cat

    Parameters
    ----------
    df : pandas.DataFrame
        A table contains many columns
    max_missing_ratio : float in [0.0,1.0], optional
        Threshold for determining a column to be too-many-missing. The default is 0.5.
    min_rows_per_value : int, optional
        Threshold for determining a column to be too-small-cat. The default is 2.
    max_ratio_per_cat : float in [0.0,1.0], optional
        Threshold for determining a column to be too-large-cat. The default is 0.9.
    verbose : bool, optional
        If True print more messages. The default is False.

    Returns
    -------
    dict
        A dictionary where a key represents a type of useless columns and
        the value is a list of useless columns of the corresponding type.

    '''
    empty_cols = []
    single_value_cols = []
    id_like_cols = []
    too_many_missing_cols = []
    too_small_cat_cols = []
    too_large_cat_cols = []

    # TODO: one-to-one map (two columns are one-to-one map), e.g. Crime 'AREA' and 'AREA NAME'
    # TODO: nearly one-to-one map  e.g. 'Premis Cd', 'Premis Desc' (both 803 and 805 are mapped to 'RETIRED (DUPLICATE) DO NOT USE THIS CODE'), rest is one-to-one)    
    # TODO: nearly one-to-one map  e.g. Crime 'Weapon Used Cd', 'Weapon Desc', 222 -> np.nan. The rest is one-to-one
    row_count = df.shape[0]
    for col in df:
        missing_count = df[col].isna().sum()
        if missing_count == row_count:
            if verbose:
                print(f'{col=} contains no value.')
            empty_cols.append(col)
            continue

        vc = df[col].value_counts(sort=True,dropna=True)
        if vc.size == 1:
            if missing_count == 0:
                if verbose:
                    print(f'{col=} contains a single value: {vc.index[0]}')
            else:
                if verbose:
                    print(f'{col=} contains a single value and missing value: {vc.index[0]}')
            single_value_cols.append(col)
            continue
        
        na_dropped = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(na_dropped):
            col_type = 'cat_like'
        elif np.array_equal(na_dropped, na_dropped.astype(int)):
            col_type = 'num_int'
        else:
            col_type = 'num_float'
            
        # a unique value for each record
        if vc.size == row_count and col_type != 'num_float': 
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} has unique value for each row')    
                id_like_cols.append(col)
                continue
            else: # col_type == 'num_int'
                print(f'warning: int column: {col} has unique value for each row.')
        
        # a unique value for each record that has value
        if vc.size + missing_count == row_count and col_type != 'num_float': 
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} has unique value for each row that has value')    
                id_like_cols.append(col)
                continue
            else: # col_type == 'num_int'
                if verbose:
                    print(f'warning: int column: {col} has unique value for each row that has value')
        
        # missing rate exceed max_missing_ratio
        missing_count = df[col].isna().sum()
        if missing_count > max_missing_ratio * row_count:
            if verbose:
                print(f'{col=} has too many missing values: {missing_count}, missing ratio > {max_missing_ratio=}')
            too_many_missing_cols.append(col)
            continue

        # too few records per category
        if vc.size > 0:
            rows_per_value = row_count / vc.size
        else:
            rows_per_value = 0
        if rows_per_value < min_rows_per_value and col_type != 'num_float':
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} rows per cat: {rows_per_value} < {min_rows_per_value=}')
                too_small_cat_cols.append(col)
                continue
            else: # col_type == 'num_int':
                if verbose:
                    print(f'warning: int column: {col} rows per cat: {rows_per_value} < {min_rows_per_value=}')
        
        max_rows_per_cat = row_count * max_ratio_per_cat
        if vc.size > 0 and vc.iloc[0] > max_rows_per_cat:
            if col_type == 'cat_like':
                if verbose:
                    print(f'cat_like column: {col} rows for largest cat {vc.index[0]}: {vc.iloc[0]} > {max_ratio_per_cat=}')
                too_large_cat_cols.append(col)
                continue
            else: # col_type == 'num_int':
                if verbose:
                    print(f'warning: int column: {col} rows for largest cat {vc.index[0]}: {vc.iloc[0]} > {max_ratio_per_cat=}')
    
    return {'empty_cols': empty_cols,
            'single_value_cols': single_value_cols,
            'id_like_cols': id_like_cols,
            'too_many_missing_cols': too_many_missing_cols,
            'too_small_cat_cols': too_small_cat_cols,
            'too_large_cat_cols': too_large_cat_cols}

def drop_useless(df, useless_cols_dict, verbose=True):
    '''
    Drop useless columns identified by find_useless_colum(df) from a dataframe df:
        drop(df, find_useless_colum(df))

    Parameters
    ----------
    df : pandas.DataFrame
        A data table.
    useless_cols_dict : dict(type,list)
        Use less columns identified by find_useless_colum(df,...) 
    verbose : bool, optional
        If true print more messages. The default is True.

    Returns
    -------
    df : pandas.DataFrame
        A copy of df with use less columns dropped.

    '''
    for useless_type in useless_cols_dict:
        cols = useless_cols_dict[useless_type]
        if verbose:
            print(f'drop {useless_type}: {cols}')
        df = df.drop(cols, axis=1)
    return df

def is_one_to_one(df, col1, col2):
    '''
    Check if col1 and col2 is one-to-one mapping

    Parameters
    ----------
    df : pandas.DataFrame
        A table
    col1 : string
        Name of a column in df
    col2 : string
        Name of a column in df

    Returns
    -------
    pd.Series
        If col1 and col2 is one-to-one mapping, return a series where index is value in col1 and value is value in col2;
        None otherwise.

    '''
    dfu = df.drop_duplicates([col1, col2])
    a = dfu.groupby(col1)[col2].count()
    b = dfu.groupby(col2)[col1].count()
    if (a.max() == 1 and a.min() == 1 and
        b.max() == 1 and b.min() == 1):
        return pd.Series(dfu[col2].values, index=dfu[col1].values)
    return None


from sklearn.preprocessing import StandardScaler
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


import warnings
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

class DataSetPreprocess:
    def __init__(self,name, df):
        '''
        df： 包含了 y, y的最后一列是 target
        '''
        useless_cols_dict = find_useless_colum(df)
        self.df = drop_useless(df, useless_cols_dict, verbose=True)
        self.name = name
        self.X = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1]
        if self.y.isnull().values.any():
            raise RuntimeError("target column has none value")
        self.y = self.y.astype(int)
        vc = self.y.value_counts(sort=True,dropna=False) # sort=True是 sort by frequencies
        if vc.size > 2:
            raise RuntimeError(f"target variable has more than 2 values: {vc}")
        if not pd.api.types.is_integer_dtype(self.y):
            raise RuntimeError("target variable is not integers") 
        
        positive = [i for i in self.y if i >0]
        negative = [i for i in self.y if i <=0]
        self.npratio = len(negative)/len(positive)
    
        
    def preprocess(self, trainTestSeedLst = [1,2,3,4,5]):
        X = self.X.copy(deep=True)
        y = self.y.copy(deep=True)
        ds_list = []
            
        #ds_list.append(("ringnorm-seed-" + str(s), X_train, y_train, X_test, y_test))
        # preprocess data 
        # for X_train
        # 1. numerical variable
        #        replace inf by max+1
        #        replace -inf by min-1
        #        filling na by mean (excluding na)
        # 2. encode categorical variable by TargetEncoding
        # for X_test, using transform
        # 3. perform min_max scaler
        for s in trainTestSeedLst:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)    
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            for col in X_train:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    col_max = X_train[col].max()
                    if col_max == np.inf:
                        warnings.warn(f'train data set: {self.name} col: {col} np.inf -> max+1')
                        new_col = X_train[col].replace([np.inf], np.nan)
                        col_max = new_col.max()
                        X_train[col].replace([np.inf], col_max+1, inplace=True)

                    col_max_test = X_test[col].max()
                    if col_max_test == np.inf:
                        warnings.warn(f'test data set: {self.name} col: {col} np.inf -> max+1')
                        X_test[col].replace([np.inf], col_max+1, inplace=True)

                    col_min = X_train[col].min()
                    if col_min == -np.inf:
                        warnings.warn(f'test data set: {self.name} col: {col} -np.inf -> min-1')
                        new_col = X_train[col].replace([-np.inf], np.nan)
                        col_min = new_col.min()
                        X_train[col].replace([-np.inf], col_min-1, inplace=True)
                    
                    col_min_test = X_test[col].min()
                    if col_min_test == -np.inf:
                        warnings.warn(f'test data set: {self.name} col: {col} -np.inf -> min-1')
                        X_test[col].replace([-np.inf], col_min-1, inplace=True)
                                                           
                    v = X_train[col].mean()
                    X_train[col] = X_train[col].fillna(v)
                    X_test[col] =  X_test[col].fillna(v)
                    
                elif pd.api.types.is_categorical_dtype(X_train[col]):
                    v = X_train[col].mode()
                    X_train[col] = X_train[col].fillna(v)
                    X_test[col] =  X_test[col].fillna(v)
                    encoder = TargetEncoder(cols=col).fit(X_train[col],y_train)
                    X_train[col] = encoder.transform(X_train[col])
                    X_test[col] = encoder.transform(X_test[col])

                    
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            ds_name = self.name + "-seed-" + str(s)
            ds_list.append((ds_name, X_train, y_train, X_test, y_test))
            
        return ds_list



if __name__ == "__main__":
    a = pd.DataFrame({'a':[1,2,np.nan,3,4], 'b':pd.Series([3,4,4,np.nan,5]).astype('category'),
                      'c':[np.nan,np.nan,np.nan,np.nan,np.nan]})
    # b = fillna(a)
    