# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:43:47 2018
1. K-means: by Euclidean distnace  
    method = {'Start':[[0,0],[1,1]]} ; initial centers
    C,U,D=kmean(data,2,method)

@author: user
"""

import numpy as np
class Clustering(object):
    def __init__(self): pass
    
    @staticmethod
    def kmean(x,k,method):
        n=np.size(x,axis=0)
        dim=np.size(x,axis=1)
        if 'Start' in method :
            Center = np.array(method['Start'],dtype=float)
        else:
            Center = np.random.random([k,dim])
        if 'maxiter' in method :
            maxiter=method['maxiter']
        else:
            maxiter=1000
        
        count_iter=0;
        while count_iter<maxiter:
            count_iter+=1
            dist=np.array(np.zeros([n,k]))
            for ic in range(k):
                c=Center[ic,:]
                dist[:,ic] = np.linalg.norm(x-c,axis=1)
            old_Center=np.copy(Center)
            Center_index=dist.argmin(axis=1)
            for ic in range(k):
                mu=np.mean(x[(Center_index==ic),:],axis=0)
                Center[ic,:]=mu
             
            th=np.linalg.norm(Center-old_Center)          
            if th<np.spacing(1):
                break
                
                
        for ic in range(k):
            c=Center[ic,:]
            dist[:,ic] = np.linalg.norm(x-c,axis=1)
        Center_index=dist.argmin(axis=1)
            
        return Center,Center_index,dist