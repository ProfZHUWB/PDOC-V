# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:05:38 2020

@author: dhlsf

贪心方法：
step 1: 找一个离边界最近的点
step 2: 找一个与1不同的最近点
step 3: 找一个与1和2不同的最近点
...
用S表示已经选中的i个点，第i+1个点需要满足的条件：
a) 离边界近
b) 与S中各点都不同--->与S中最近点距离比较大
"""

from scipy.spatial import distance
import numpy as np

from proximity import addProximity



def selectTopkFast(origP, origPW, k, mdWeight = 1.0, dist=distance.euclidean):
    '''
    通过贪心的算法在origP中找出top K个点，使得这 K 个点离边界近，同时与S中各点不相同

    Parameters
    ----------
    origP : numpy.array
        origP[i]: 正样本点中第 i 个正样本点
    
    origPW : numpy.array
        origPW[i]: 正样本点中第 i 个正样本点离边界的proximity
    
    k : int
        需要从正样本中挑选符合要求(离边界近，同时又与S中各点不相同)的点的个数.
        
    mdWeight : TYPE, optional
        diversity需要的权重. The default is 1.0. 
        mdWeight = w*weightP
    dist : TYPE, optional
        DESCRIPTION. The default is distance.euclidean.

    Returns
    -------
    TYPE
        DESCRIPTION.
    sPW : TYPE
        sPW[i] 记录 S[i] 对应的 proximity + minDist.
    sI : TYPE
        DESCRIPTION.

    '''
    plen = len(origP)
    P = np.copy(origP)
    PW = np.copy(origPW)
    idx = [i for i in range(plen)]  # idx[i] 是 P[i] 在origP 中的下标

    # # debug    
    # print("========== begin =============")
    # print("P: ",P)
    # print("PW: ",PW)
    # print("idx: ",idx)
    # print("plen: ",plen)
        
    # 找第一个点
    maxp = np.argmax(PW)
    
    # add p to S
    point = np.copy(P[maxp])
    #S = [np.copy(point)]   # S 记录top K个点; S = [point] 如果一个点超过1维，是用数组保存，point是数组的地址，后来P[maxp] = P[plen] 会替换掉它
    sPW = np.zeros(k) # sPW[i] 记录 S[i] 对应的 proximity + minDist
    sPW[0] = PW[maxp]
    sI = [maxp]   # 记录选择的 k 个点的原始下标, 一开始 idx[maxp] 就是 maxp
    
    # 删除第一个点
    plen -= 1
    P[maxp] = P[plen]
    PW[maxp] = PW[plen]
    idx[maxp] = idx[plen]
        
    # md[i] 是 P[i] 到 S 的最小距离， S 目前只有一个点
    md = np.zeros(plen)
    for j in range(plen):
        md[j] = dist(point, P[j])

    import time
    nextPrintTime = time.time()
        
    for i in range(1,k):
        if time.time() > nextPrintTime:
            nextPrintTime += 30
            print("select {0:6.0f}/{1:6.0f}".format(i,k))
        
        # 找到第 i 个点
        maxp = np.argmax(mdWeight*md[0:plen]+PW[0:plen])
        
        # add p to S
        point = np.copy(P[maxp])
        #S.append(np.copy(point)) # 错误 S.append(point)
        sPW[i] = PW[maxp]
        sI.append(idx[maxp])
        
        # remove p from P
        plen -= 1
        P[maxp] = P[plen]
        PW[maxp] = PW[plen]
        md[maxp] = md[plen]
        idx[maxp] = idx[plen]

        # 当 S' = S + {point} 时 md'[i] 要用新加入的点 point 更新
        for j in range(plen):
            md[j] = min(md[j], dist(point, P[j]))

        # # debug    
        # print("------ loop ---------, i: ",i, " maxp: ",maxp)
        # print("P: ",P[0:plen])
        # print("PW: ",PW[0:plen])
        # print("idx: ",idx[0:plen])
        # print("md: ",md[0:plen])
        # print("plen: ",plen)
        # print("sI: ",sI)
        
    return origP[sI],sPW,sI



def selectTopKAsDS(ts, p_count, n_count, p, k, w,dist=distance.euclidean):
    '''
    根据 Proximity (p, k) + w * Diversity 从 ts 的正负样本中各挑选 pratio, nratio 数据
    构成一个新的数据集

    Parameters
    ----------
    ts : 一个数据集
        DESCRIPTION.
    p_count : int
        挑选正样本的个数.
    n_count : int
        挑选负样本的个数.
    p : 计算 Proximity 的方法
        DESCRIPTION.
    k : Proximity 中的参数 k
        DESCRIPTION.
    w : Diversity 相对与 Proxmity 的权重
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    p_count = min(p_count, len(ts.P))
    n_count = min(n_count, len(ts.N))
    
    nds = addProximity(ts, k = k, clossnessCalc=p,dist = dist)
    if w == 0:
        return nds.getSubset(p_count,n_count)
    else:
        raise RuntimeError('w != 0 not implemented yet')
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

