# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:44:15 2021
所有数据集每个算法都用 grid search 搜索一个最好的参数
筛选数据集的方法：
1. AC
2. AC + 2*diversity
3. OutDist
4. OutDist + 2*diversity
5. Freq
6. Freq + 2*diversity
7. Random Selection
8. Select by Voronoi
@author: dhlsf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
from voronoi_explore import yTransform


from training_util import savePkl,loadPkl

def linkCount(num_col, layer_size):
    '''
    Compute number of links in an feedforward neural network (FNN)
    
    Parameters
    ----------
    num_col : int
        number of input variables (columns in training data matrix X)
    layer_size : list of int
        number of neurals in each layer

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    count : int
        number of links (trainable weights) in a neural network

    '''
    if len(layer_size) < 1:
        raise RuntimeError
        
    count = num_col * layer_size[0]
    for i in range(1,len(layer_size)):
        count += layer_size[i-1] * layer_size[i]
    count += layer_size[-1]
    return count


allModelNames = ['NeuralNet','GBDT','RBFSVM','DT','kNN', 'RF']                                    


# 实验用的模型列表： [(模型名字、模型、模型参数范围), ..., ]
# 有些模型的参数范围与数据集大小、数据的范围有关
def prepareModelList(X_train, y_train, modelNames):
    # 根据训练样本数，为 kNN 设置合理的 k
    (num_row, num_col) = X_train.shape
    if num_row >= 20:
        kLst = [5,10,20]
    elif num_row >= 10:
        kLst = [5,10]
    elif num_row >= 5:
        kLst = [5]
    else:
        kLst = [1]        
    
    # 神经网络中的可训练权重数应该小于样本数的 1/3, 据此选择适当的 2层，3层网络
    hidden_layer_list = [(8,)]
    for layer_size in [(16,),(32,),(64,),(128,),(256,),(512,),(16,16),(16,32),(32,32),(32,64),(64,64)]:
        link_count = linkCount(num_col, layer_size)
        if link_count * 3 <= num_row:
            hidden_layer_list.append(layer_size)    
    
    models = []
    for name in modelNames:
        if name == 'NeuralNet':
            models.append(
                ('NeuralNet', MLPClassifier(learning_rate = 'adaptive', random_state=1,
                                            max_iter=1000, early_stopping = False), {
                    'activation'         : ['logistic', 'tanh', 'relu'],
                    'solver'             : ['sgd', 'adam'],
                    'hidden_layer_sizes' : hidden_layer_list
                }))
                                   
        elif name == 'GBDT':
            models.append(                                        
                ('GBDT', GradientBoostingClassifier(), {
                    'n_estimators'     : [8,16,32,64],
                    'learning_rate'    : [0.001,0.01,0.1]
                }))
        elif name == 'RBFSVM':
            models.append(                                                
                ('RBFSVM', SVC(probability=True), {
                    'gamma'        : [0.01,0.1,1,10],
                    'C'            : [0.1,1,10,100],
                    'class_weight' : [None, 'balanced']}
                  ))
        elif name == 'DT':
            models.append(                                                
                ('DT', DecisionTreeClassifier(), {
                    'max_depth'        : [5,10,15],
                    'min_samples_leaf' : [1,5,10,15],
                    'class_weight'     : [None, 'balanced']
                }))
        elif name == 'kNN':
            models.append(                                                
                ('kNN', KNeighborsClassifier(), {
                    'n_neighbors' : kLst,
                    'weights'     : ['uniform', 'distance']
                }))
        elif name == 'RF':
            models.append(                                                
                ('RF', RandomForestClassifier(), {
                    'n_estimators'     : [8,16,32,64],
                    'max_features'     : ['sqrt', 'log2', None],
                    'max_depth'        : [5,10,15], 
                    'min_samples_leaf' : [1,5,10,15],
                    'class_weight'     : [None, 'balanced']
                }))
        else:
            raise RuntimeError(f" {name} is not specificed")
    
    return models
                                 

def run_models(oriXTrain, oriYTrain, X_train, y_train, X_test, y_test, strategyName, ds_name, selRatio,
               modelNames, dir_name = None, cv=5, summary_filename=None,
               **grid_kwargs):
    '''
    Run all models on a data set

    Parameters
    ----------
    oriXTrain: np.array((num_row,now_col))
        initial TrainX, without any selection
    oriYTrain: np.array((num_row,now_col))
        initial TrainY, without any selection    
    X_train : np.array((num_row,now_col))
        Training data matrix
    y_train : np.array((num_row,))
        Training data label
    X_test : np.array((num_row,now_col))
        Test data matrix
    y_test : np.array((num_row,))
        Test data label
    cv : int, optional
        Cross validation folds. The default is 5.
    ds_name : string
        Name of data set
    modelLst : List[(name,model,params)]
        List of models, each mode is a tuple consists of name, model, params to be searched in GridSearchCV
    dir_name : string, optional
        If not None:
            1) plot the data set to {ds_name}.png if num_col = 2
            2) write training log to {ds_name}-log.txt
            3) write detailed cross validation results to {ds_name}-cv.csv                                
    **grid_kwargs : TYPE
        Other parameters to be passed to GridSearchCV

    Returns
    -------
    exp_result : Dict(model_name, model_result)
        A dictionary that records model_result for each model name.
        Model_result is again a dictionary：
            'acc': accuracy of a model on test set
            'f1':  F1 score of a mdoel on test set
            'time': total time in seconds for GridSearch and final fitting
            'grid_search': the grid_search result returned by GridSearchCV
    '''
    oriYTrain = yTransform(oriYTrain)
    y_train = yTransform(y_train)
    y_test = yTransform(y_test)

    if (oriXTrain.shape[1] != X_train.shape[1]) or (oriXTrain.shape[1] != X_test.shape[1]) or (X_train.shape[1]!= X_test.shape[1]):
        raise RuntimeError

    log_file = None
    if dir_name is not None:    
        os.makedirs(dir_name, exist_ok=True)
        log_file = open(dir_name + '/' + ds_name + '-log.txt', 'w')

        
    modelLst = prepareModelList(X_train, y_train, modelNames=modelNames)

    for (key,model,params) in modelLst:
        if log_file is not None:
            log_file.write(f"Running GridSearchCV for {key}\n")
            log_file.write(f"{X_train.shape = },{y_train.shape = },{X_test.shape = },{y_test.shape = }\n")
            log_file.flush()
        # print(f"Running GridSearchCV for {key}")

        model_grid_search_result = dir_name+'/grid_search-'+key+".pkl"
        if os.path.exists(model_grid_search_result):
            print(f'{key=} already executed, skip')
        
               
        timeStart = time.process_time()
        grid_search = GridSearchCV(model, params, cv=cv, scoring='accuracy', return_train_score = True, **grid_kwargs)
        grid_search.fit(X_train, y_train)
        if log_file is not None:
            log_file.write(f"  **best parameters = {grid_search.best_params_}\n")
            log_file.flush()
        # print(f"  **best parameters = {grid_search.best_params_}")
        
        y_predict = grid_search.predict(X_test)  # Call predict on the estimator with the best found parameters
        y_predict_prob = grid_search.predict_proba(X_test)
        accuracy = accuracy_score(y_test,y_predict)
        f1 = f1_score(y_test,y_predict)
        
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
        # auc = metrics.auc(fpr, tpr)
        # probability is needed while calculating auc
        auc = metrics.roc_auc_score(y_test, y_predict_prob[:, 1])
        
        t = time.process_time() - timeStart
        
        if log_file is not None:
            log_file.write(f"  **time: {t:.4f}\n")
            log_file.write(f"  **Accuracy of the best classifier after 5 CV is {accuracy:.4f}\n")
            log_file.write(f"  **f1 Score of the best classifier after 5 CV is {f1:.4f}\n")
            log_file.write(f"  **auc Score of the best classifier after 5 CV is {auc:.4f}\n")            
            log_file.write("----------------------\n")
            log_file.flush()

        with open(summary_filename, 'a+') as f:
            if os.stat(summary_filename).st_size == 0:
                f.write("method,ds-name,model,orgTrainRow,dimention,testRow,selTrainRow,selRatio,measure,value\n") #,model,performance
                f.flush()      
            
            f.write(f"{strategyName},{ds_name},{key},{oriXTrain.shape[0]},{oriXTrain.shape[1]},{X_test.shape[0]},{X_train.shape[0]},{selRatio},acc,{accuracy}\n")
            f.write(f"{strategyName},{ds_name},{key},{oriXTrain.shape[0]},{oriXTrain.shape[1]},{X_test.shape[0]},{X_train.shape[0]},{selRatio},f1,{f1}\n")
            f.write(f"{strategyName},{ds_name},{key},{oriXTrain.shape[0]},{oriXTrain.shape[1]},{X_test.shape[0]},{X_train.shape[0]},{selRatio},auc,{auc}\n")
            f.write(f"{strategyName},{ds_name},{key},{oriXTrain.shape[0]},{oriXTrain.shape[1]},{X_test.shape[0]},{X_train.shape[0]},{selRatio},time(s),{t}\n")
            f.flush()                  
            
        if dir_name is not None:
            savePkl(grid_search, model_grid_search_result)
        # print(f"  **time: {time.time() - timeStart}")
        # print(f"  **Accuracy of the best classifier after 5 CV is {accuracy:.4f}")
        # print(f"  **f1 Score of the best classifier after 5 CV is {f1:.4f}")
        # print(f"  **auc Score of the best classifier after 5 CV is {auc:.4f}")
        # print("----------------------")
    
    # if dir_name is not None:
    #     frames = []
    #     for name in exp_result:
    #         grid_search = exp_result[name]['grid_search']
    #         frame = pd.DataFrame(grid_search.cv_results_)
    #         frame = frame.filter(regex='^(?!.*param_).*$')
    #         frame['estimator'] = len(frame)*[name]
    #         frames.append(frame)
    #     df = pd.concat(frames)
        
    # #   df = df.sort_values([sort_by], ascending=False)
    # #   df = df.reset_index()
    #     # df = df.drop(['rank_test_score', 'index'], 1)
    #     df = df.drop(['rank_test_score'], 1)
        
    #     columns = df.columns.tolist()
    #     columns.remove('estimator')
    #     columns = ['estimator']+columns
    #     df = df[columns]
    #     df.to_csv(dir_name+'/'+ds_name+'-cv.csv', index=False)

 




import dataset as ds
import proximity as pr
import selectTopK as dr 
from scipy.spatial import distance

def selsectByProximityAC(ts,p_count,n_count,w,k=20,dist = distance.euclidean):
    return dr.selectTopKAsDS(ts = ts, p_count = p_count, n_count = n_count, 
                            p = pr.computeClosenessToMarginByAlternatingChain, k = k, w = w,dist=dist)

def selectByProximityFrequencyInOppKNN(ts,p_count,n_count,w, k=20,dist = distance.euclidean):
    return dr.selectTopKAsDS(ts = ts, p_count = p_count, n_count = n_count, 
                            p = pr.computeClosenessToMarginByFreqeucyInOppKNN, k = k, w = w,dist=dist)

def selectByProximityEntropy(ts,p_count,n_count,w,k=20,dist = distance.euclidean):
    return dr.selectTopKAsDS(ts = ts, p_count = p_count, n_count = n_count, 
                            p = pr.computeClosenessToMarginByEntropy, k = k, w = w,dist=dist)

    
# 1. select a subset by AC
def byAC(ts,p_count,n_count,*args, **kwargs):
    return selsectByProximityAC(ts,p_count,n_count,w=0)
    
# 5. select a subset by Freq
def byFreq(ts,p_count,n_count,*args, **kwargs):
    return selectByProximityFrequencyInOppKNN(ts,p_count,n_count,w=0)
    

from voronoi_explore import remove_noise_by_voronoi

# 10. select a subset by Voronoi with filter 0.2
def byVoronoiFilter02(ts,p_count,n_count,*args, **kwargs):
    ds_v = remove_noise_by_voronoi(ts,min_dist_to_boundary=0.2)
    return ds_v.getSubset(p_count,n_count)

# 13. select a subset by Entropy k=20
def byEntropy20(ts,p_count,n_count,*args, **kwargs):
    return selectByProximityEntropy(ts,p_count,n_count,w=0,k=20)

from lsh_methods import LSH_IF_F_binary_search

def byLSH_IS_F_bs(ts,p_count,n_count,w_alpha=0.1,*args, **kwargs):
    selected_idx,w = LSH_IF_F_binary_search(ts, alpha=w_alpha)
    X,y = ts.toXY()
    if not isinstance(selected_idx, np.ndarray):
        selected_idx = np.array(selected_idx)
    selected_idx = selected_idx.astype(int)            
    X_selected_sub = X[selected_idx]
    y_selected_sub = y[selected_idx]
    return ds.TrainSet.gen(X_selected_sub, y_selected_sub, ts.dsDir+f'/LSH_IS_F_bs={w_alpha}')
    
allStrategyList = [
    byAC, # ENNC 
    byFreq, # NC
    byVoronoiFilter02, # PDOC-V
    byEntropy20, # NE
    byLSH_IS_F_bs] # LSH


def exp_selection_methods(ds_list, data_Dir = "../result/data_knn", 
                          modelNames = allModelNames,
                          strategyList = allStrategyList,
                          selRatioLst = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1], 
                          min_samples_in_class = 30,
                          dir_name = '../result/exp-2021-10-14'):

    os.makedirs(dir_name,exist_ok=True)
    overlap_avgDict = {}

    start_time = time.time()    
    for i,(ds_name, X_train, y_train, X_test, y_test) in enumerate(ds_list):
        print(f'{i=}: {ds_name=}, time: {time.time()-start_time:.2f} s')
        
        dsDir = os.path.join(data_Dir, ds_name)
        os.makedirs(dsDir,exist_ok=True)
        ts = ds.TrainSet.gen(X_train,y_train,dsDir)
        summary_filename = dir_name + '/summary.csv'
        ds_dir = dir_name + '/' + ds_name
        os.makedirs(ds_dir,exist_ok=True)
            
        for selRatio in selRatioLst:
            p_count = int(selRatio * len(ts.P)+0.5)
            n_count = int(selRatio * len(ts.N)+0.5)
            if p_count < min_samples_in_class or n_count < min_samples_in_class:
                continue

            print(f'    {selRatio=}, time: {time.time()-start_time:.2f} s')
            sel_dir_name = f'{ds_dir}/{selRatio=}'
            os.makedirs(sel_dir_name,exist_ok=True)
            
            # Select by strategy
            if len(strategyList) <= 0:
                continue
                
            for strategy in strategyList:
                strategyName = strategy.__name__
                print(f'        {strategyName=}, time: {time.time()-start_time:.2f} s')
                new_ds_name = strategyName + "-" + ds_name
                
                sub_fname = sel_dir_name+'/'+strategyName+'.pkl'
                if os.path.exists(sub_fname):
                    print(f'load selected dataset {sub_fname=}')
                    sub = loadPkl(sub_fname)
                else:
                    sub = strategy(ts = ts, p_count = p_count, n_count = n_count, w_alpha=selRatio)
                    savePkl(sub, sub_fname)
                    print(f'save selected dataset {sub_fname=}')

                    if ts.P.shape[1] == 2:
                        sub.plot_selected_2D(ts,s=50)
                        plt.title(f'{new_ds_name}')
                        plt.savefig(sel_dir_name + '/' + f"{strategyName}"+".png")
                        plt.show()
                
                if len(sub.P) < min_samples_in_class or len(sub.N) < min_samples_in_class:
                    print(f'too few instances {len(sub.P)=}, {len(sub.N)=}, skip {selRatio=}')
                    continue
                
                if hasattr(sub,'avg_overlap'):
                    overlap_avgDict[ds_name] = sub.avg_overlap

                X_selected_sub, y_selected_sub = sub.toXY() 
                run_models(X_train,y_train,X_selected_sub,y_selected_sub,X_test, y_test, strategyName, ds_name, selRatio, modelNames, dir_name = sel_dir_name+'/'+strategyName, summary_filename=summary_filename)                    
                                
            new_ds_name = 'no-selection-' + ds_name
            run_models(X_train,y_train,X_train,y_train,X_test, y_test, 'noselect', ds_name, 1.0, modelNames, dir_name = ds_dir+'/no-selection', summary_filename=summary_filename)                    
    
    savePkl(overlap_avgDict,dir_name+"/overlap_avgDict.pkl")


###################### 2022-05-20 ############################    

exp_modelNames = ['NeuralNet',  # MLP
                  'GBDT',
                  'RBFSVM',     # SVM
                  'DT',
                  'kNN',
                  'RF']     
exp_strategy_list = [byLSH_IS_F_bs,     # LSH
                     byAC,              # ENNC
                     byFreq,            # NC
                     byVoronoiFilter02, # PDOC-V
                     byEntropy20]       # NE

def load_natural_datasets():
    # natural dataset
    from training_util import load_banana_dataset, load_ringnorm_dataset,load_twonorm_dataset
    from training_util import load_employee_salaries_dataset, load_banknote_authentication_dataset
    from training_util import load_colleges_dataset,load_url_spam_dataset,load_Eeg_eye_state_dataset,load_mushroom_dataset
    ds_list_all = load_banana_dataset() + load_ringnorm_dataset() + load_twonorm_dataset()\
                + load_employee_salaries_dataset()\
                + load_banknote_authentication_dataset()\
                + load_colleges_dataset() + load_url_spam_dataset() + load_Eeg_eye_state_dataset()\
                + load_mushroom_dataset()
    return ds_list_all


    
if __name__ == '__main__': 

    
    # synthetic dataset
    from training_util import genSyntheticDataSets
    exp_selection_methods(genSyntheticDataSets(), 
                          data_Dir = "../result/data_knn", 
                          modelNames = exp_modelNames,
                          strategyList = exp_strategy_list,
                          selRatioLst = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1], 
                          min_samples_in_class = 30,
                          dir_name = '../result/exp-2022-05-20/synthetic')

                    
    exp_selection_methods(load_natural_datasets(), 
                          data_Dir = "../result/data_knn", 
                          modelNames = exp_modelNames,
                          strategyList = exp_strategy_list,
                          selRatioLst = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1], 
                          min_samples_in_class = 30,
                          dir_name = '../result/exp-2022-05-20/natural')          