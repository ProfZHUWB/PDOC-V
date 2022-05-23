# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 20:37:07 2022

@author: YingFu
"""
import os
import pickle

def savePkl(obj, fileName):
    dirName = os.path.dirname(fileName)
    if len(dirName) > 0:
        os.makedirs(dirName, exist_ok=True)
    with open(fileName, "wb") as file:
        pickle.dump(obj,file)

def loadPkl(fileName):
    with open(fileName, "rb") as file:
        return pickle.load(file)
    
    
from sklearn.preprocessing import StandardScaler
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


import numpy as np
import pandas as pd


def csvDataLoad(fileName):
    df = pd.read_csv("../data/" + fileName)
    data = df.values
    y = data[:,-1]
    X = data[:,:-1]
    return X,y

from sklearn.datasets import make_moons, make_circles, make_blobs
def genMoon(n_samples, noise, trainRandom, testRandom):
    X_train, y_train = make_moons(n_samples = n_samples, noise = noise, random_state = trainRandom)
    X_test, y_test = make_moons(n_samples = n_samples, noise = noise, random_state = testRandom)    
    y_train.astype(int)
    y_test.astype(int)
    return X_train, y_train, X_test, y_test

def genCircle(n_samples, noise, trainRandom, testRandom):
    X_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=trainRandom)
    X_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=testRandom)
    return X_train, y_train, X_test, y_test

def genBlobs(n_samples, noise, trainRandom, testRandom):
    X_train, y_train = make_blobs(n_samples=n_samples, n_features=2, centers=[[-1,0],[1,0]], cluster_std=noise, random_state = trainRandom)
    X_test, y_test = make_blobs(n_samples=n_samples, n_features=2, centers=[[-1,0],[1,0]], cluster_std=noise, random_state = testRandom)
    return X_train, y_train, X_test, y_test

def genXOR(random_state=4, sigma=0.8, Eachn_samples=250):        
    np.random.seed(random_state)        
    def gen_clusters(cx1,cx2):
        mean = [cx1,cx2]
        cov = [[sigma * sigma,0],[0,sigma*sigma]]
        data = np.random.multivariate_normal(mean,cov,Eachn_samples)
        return data
    
    data_p1 = gen_clusters(1,1)
    data_p2 = gen_clusters(-1,-1)
    data_n1 = gen_clusters(1,-1)
    data_n2 = gen_clusters(-1,1)

    P = np.vstack((data_p1,data_p2))
    N = np.vstack((data_n1,data_n2))
    y1 = np.array([1 for i in range(len(P))])
    y2 = np.array([0 for i in range(len(N))])
    
    X = np.vstack((P,N))
    y = np.hstack((y1,y2))
    
    return X,y    

def genxor(n_samples, noise, trainRandom, testRandom):
    X_train, y_train = genXOR(sigma = noise,Eachn_samples = int(n_samples/4), random_state = trainRandom)                
    X_test, y_test = genXOR(sigma = noise,Eachn_samples = int(n_samples/4),random_state = testRandom)
    return X_train, y_train, X_test, y_test


def genSyntheticDataSets(trainRandomLst = [1,2,3,4,5],noiseLst = [0.2, 0.8, 1.5],n_samples = 5000):
    '''
    人工生成的，不同重合程度的数据集
    '''
    testRandom = 0
    ds_list = []
    
    for noise in noiseLst:
        for trainRandom in trainRandomLst:
            dsName = "moon-noise-" + str(noise) + "-Trainseed-" + str(trainRandom) 
            X_train, y_train, X_test, y_test = genMoon(n_samples, noise, trainRandom, testRandom)
            ds_list.append((dsName,X_train,y_train,X_test,y_test))

        for trainRandom in trainRandomLst:
            dsName = "circle-noise-" + str(noise) + "-Trainseed-" + str(trainRandom)
            X_train, y_train, X_test, y_test = genCircle(n_samples, noise, trainRandom, testRandom)
            ds_list.append((dsName,X_train,y_train,X_test,y_test))
        
        for trainRandom in trainRandomLst:
            dsName = "blobs-noise-" + str(noise) + "-Trainseed-" + str(trainRandom)
            X_train, y_train, X_test, y_test = genBlobs(n_samples, noise, trainRandom, testRandom)
            ds_list.append((dsName,X_train,y_train,X_test,y_test))
                
        for trainRandom in trainRandomLst:
            dsName = "XOR-noise-" + str(noise) + "-Trainseed-" + str(trainRandom)
            X_train, y_train, X_test, y_test = genxor(n_samples, noise, trainRandom, testRandom)
            ds_list.append((dsName,X_train,y_train,X_test,y_test))
    
    return ds_list


from sklearn.model_selection import train_test_split
from data_preprocess import url_spam,mushroom,misc_colleges,employee_salaries,uci_adult

# 1. banana
def load_banana_dataset(trainTestSeedLst = [1,2,3,4,5]):
    ds_list = []
    X,y = csvDataLoad("banana/banana.csv")
    for s in trainTestSeedLst:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        X_train, X_test = scale_data(X_train, X_test)
        ds_list.append(("banana-seed-" + str(s), X_train, y_train, X_test, y_test))
    return ds_list

# 2. breast cancer
def load_breast_cancer_dataset(trainTestSeedLst = [1,2,3,4,5]):
    ds_list = []
    from sklearn.datasets import load_breast_cancer
    X = load_breast_cancer().data
    y = load_breast_cancer().target
    for s in trainTestSeedLst:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        X_train, X_test = scale_data(X_train, X_test)
        ds_list.append(("breastCancer-seed-" + str(s), X_train, y_train, X_test, y_test))
    return ds_list

# 3. BankNote_authentication
def load_banknote_authentication_dataset(trainTestSeedLst = [1,2,3,4,5]):
    ds_list = []
    X,y = csvDataLoad("BankNote_Authentication/BankNote_Authentication.csv")
    for s in trainTestSeedLst:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        X_train, X_test = scale_data(X_train, X_test)
        ds_list.append(("BankNote_Authentication-seed-" + str(s), X_train, y_train, X_test, y_test))
    return ds_list


# 4. ringnorm
def load_ringnorm_dataset(trainTestSeedLst = [1,2,3,4,5]):
    ds_list = []
    X,y = csvDataLoad("ringnorm/ringnorm.csv")
    for s in trainTestSeedLst:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        X_train, X_test = scale_data(X_train, X_test)
        ds_list.append(("ringnorm-seed-" + str(s), X_train, y_train, X_test, y_test))
    return ds_list


# 5. twonorm
def load_twonorm_dataset(trainTestSeedLst = [1,2,3,4,5]):
    ds_list = []
    X,y = csvDataLoad("twonorm/twonorm.csv")
    for s in trainTestSeedLst:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        X_train, X_test = scale_data(X_train, X_test)
        ds_list.append(("twonorm-seed-" + str(s), X_train, y_train, X_test, y_test))
    return ds_list

# 6. eeg-eye-state
def load_Eeg_eye_state_dataset(trainTestSeedLst = [1,2,3,4,5]):
    ds_list = []
    X,y = csvDataLoad("eeg-eye-state/eeg-eye-state.csv")
    for s in trainTestSeedLst:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        X_train, X_test = scale_data(X_train, X_test)
        ds_list.append(("eeg-eye-state-seed-" + str(s), X_train, y_train, X_test, y_test))
    return ds_list


# 7. employee_salaries
def load_employee_salaries_dataset(trainTestSeedLst = [1,2,3,4,5]):
    return employee_salaries.load("../data",trainTestSeedLst = trainTestSeedLst)


# 8. colleges
def load_colleges_dataset(trainTestSeedLst = [1,2,3,4,5]):
    return misc_colleges.load("../data",trainTestSeedLst = trainTestSeedLst)


# 9. url_spam
def load_url_spam_dataset(trainTestSeedLst = [1,2,3,4,5]):
    return url_spam.load("../data",trainTestSeedLst = trainTestSeedLst)


# 10. mushroom
def load_mushroom_dataset(trainTestSeedLst = [1,2,3,4,5]):
    return mushroom.load("../data",trainTestSeedLst = trainTestSeedLst)

# 11. adult
def load_adult_dataset(trainTestSeedLst = [1,2,3,4,5]):
    return uci_adult.load("../data",trainTestSeedLst = trainTestSeedLst)



if __name__ == '__main__': 
    # data_dir="../data"
    # ds_list_url_spam = url_spam.load(data_dir,take_sample=True)
    # ds_list_mushroom = load_mushroom_dataset(trainTestSeedLst = [4,5])
    ds_list_adult = load_adult_dataset(trainTestSeedLst = [1,2,3,4,5])



