import numpy as np
import random
import math

def randperm(a):
    b = []
    while(1):
        r = random.choice(a)
        r=np.where(a==r)[0][0]
        b.append(a[r])
        a=np.delete(a, r, 0)
        if a.size==0:
            break
    b=np.array(b)
    return b

def crossvalind_kfold(label,k):
    N=len(label)
    GroupLabel=np.zeros([N,2],dtype=int)
    GroupLabel[:,0]=label
    TrueLabel=np.unique(label)
    for i in TrueLabel:
        tmp=np.where(label==i)[0]
        rand_tmp=randperm(tmp)
        n_tmp=len(tmp)
        sub_n=int(np.floor(n_tmp/k))
        if sub_n==0:
            for j in range(n_tmp):
                GroupLabel[rand_tmp[j-1],[1]]=int(j+1)  
        else:
            for j in range(k):
                t1=j*sub_n
                t2=t1+sub_n
                if j==k-1:
                    t2=n_tmp
                    
                if t1==t2:
                    pos=0;
                else:
                    pos=np.array(range(t1,t2))
                GroupLabel[rand_tmp[pos],[1]]=int(j+1)
    return GroupLabel 

def crossvalind_kfold_regression(label,k):
    N=len(label)
    GroupLabel=np.zeros([N,2],dtype=float)
    GroupLabel[:,0]=label
    tmp=np.array(range(N))
    rand_tmp=randperm(tmp)
    sub_n=int(np.floor(N/k))
    for j in range(k):
        t1=j*sub_n
        t2=t1+sub_n-1
        if j==(k-1):
            t2=N-1      
        pos=np.array(range(t1,t2));
        GroupLabel[rand_tmp[pos],[1]]=j
    return GroupLabel  

def VIIndex(TrueLabel, PredLabel): 
    N=len(TrueLabel)  
    trueIndex=np.unique(TrueLabel)
    predIndex=np.unique(PredLabel)
    shapesize=max([max(trueIndex),max(predIndex)])+1
    C=np.zeros([int(shapesize),int(shapesize)])
    Index_row,Index_cloumn=[],[]
    for i,icontext in enumerate(trueIndex):
        Index_row.append(icontext)
        for j,jcontext in enumerate(predIndex):
            tmp=np.where((TrueLabel==icontext)&(PredLabel==jcontext))[0]
            C[icontext,jcontext]=len(tmp)
            Index_cloumn.append(jcontext)
    acc=np.sum(np.diag(C))/N
    acc_perclass=np.diag(C)/np.sum(C,axis=1)
    
    
    NormalziedC=C/N;
    p0=np.sum(np.diag(NormalziedC));
    c1=np.sum(NormalziedC,axis=0);
    r=np.sum(NormalziedC,axis=1);
    pc=np.sum(c1*r);
    kappa=(p0-pc)/(1-pc);     
    
              
    return C,acc, acc_perclass ,kappa

def RMSE(pred_DATA, ture_DATA):
    n = ture_DATA.size
    count = 0
    for i in range(n):
        count = count + math.pow((ture_DATA[i] - pred_DATA[i]), 2)
    result = math.sqrt(count / n)
    
    return result