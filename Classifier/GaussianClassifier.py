import numpy as np
from numpy import matlib


class GaussianClassifier(object):
    def __init__(self, data, label,testdata):
        self.data=data
        self.label=label
        self.testdata=testdata
        
    def GaussianClassifier_train(self):
        data=self.data
        label=self.label
        mu=[]
        sigma=[]
        TL=np.unique(label)
        for item in TL:
            pos=np.where(label==item)
            p1=pos[0]
            mu.append(np.average(data[p1], axis=0))
            sigma.append(np.cov(data[p1].T))
        self.mu=mu
        self.sigma=sigma
    
    def GaussianClassifier_test(self):
        testdata=self.testdata
        label=self.label
        mu=self.mu
        sigma=self.sigma
        Ny,dim=testdata.shape
        TL=np.unique(label)
        p=(np.zeros([dim,Ny]))

        for i in range(len(TL)):
            m=mu[i]
            s=sigma[i]
            tmpm=np.subtract(testdata,matlib.repmat(m, Ny, 1))
            tmps=np.linalg.pinv(s)
            tmp=np.dot(np.dot(tmpm,tmps),np.transpose(tmpm))
            dy=np.diag(tmp)
            tmp=np.exp(-dy/2)  
            tmp=tmp*(np.linalg.det(s)**(-0.5))*((2*np.pi)**(-dim/2))
            p[:][i]=(tmp)
        p1=(np.zeros([1,Ny]))
        p1=matlib.repmat(np.sum(p, axis=0),dim,1)
        p=p/p1       
        p=np.transpose(p)
        return p