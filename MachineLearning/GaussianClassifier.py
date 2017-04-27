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
        '''When yor run a larger number of testing data, the out-of-memory error may occur.
            Hence, we have to do a partial segmentation for the testing data.
            In my code, if the number of testing data is larger than 10,000, 
               the mechanism of partial segmentation would work. 
        '''
        if Ny>10000: 
            P=(np.zeros([Ny,dim])) 
            epoch=int(np.floor(Ny/10000))
            for epo in range(epoch):
                index=range(0+epo*10000,(1+epo)*10000)
                tmptestdata=testdata[index,:]
                pp=GaussianClassifier.__GaussianClassifier_test_calculatProbability(tmptestdata,mu,sigma,TL) 
                P[index,:]=pp
            index=range(index[-1],Ny)
            tmptestdata=testdata[index,:]
            pp=GaussianClassifier.__GaussianClassifier_test_calculatProbability(tmptestdata,mu,sigma,TL) 
            P[index,:]=pp                
        else:
            P=GaussianClassifier.__GaussianClassifier_test_calculatProbability(testdata,mu,sigma,TL)    
        return P
    
    def __GaussianClassifier_test_calculatProbability(testdata,mu,sigma,TL):
        Ny,dim=testdata.shape
        P=(np.zeros([dim,Ny]))
        for i in range(len(TL)):
            m=mu[i]
            s=sigma[i]
            tmpm=np.subtract(testdata,matlib.repmat(m, Ny, 1))
            tmps=np.linalg.pinv(s)
            tmp=np.dot(tmpm,tmps)
            tmp=np.dot(tmp,np.transpose(tmpm))
            dy=np.diag(tmp)
            tmp=np.exp(-dy/2)  
            tmp=tmp*(np.linalg.det(s)**(-0.5))*((2*np.pi)**(-dim/2))
            P[:][i]=(tmp)
        p1=(np.zeros([1,Ny]))
        p1=matlib.repmat(np.sum(P, axis=0),dim,1)
        P=P/p1       
        P=np.transpose(P)
        return P