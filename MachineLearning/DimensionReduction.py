import numpy as np
from numpy import matlib
class DimensionReduction(object):
    def __init__(self): pass
        
    def PCA(self,data):
        Sigma=DimensionReduction.__get_CovarianceMatrix(data)
        RotMatrix, coe_PC, U=np.linalg.svd(Sigma)
        xRot = np.dot(RotMatrix,np.transpose(data))
        xRot = np.transpose(xRot)                        
        return coe_PC, xRot
    
    def PCA_Whitening(self,data):
        Sigma=DimensionReduction.__get_CovarianceMatrix(data)
        RotMatrix, coe_PC, U=np.linalg.svd(Sigma)
        
        epsilon=10**(-5);
        tmp=np.diag(1/np.sqrt(coe_PC+epsilon))
        xPCAwhite = np.dot(tmp,np.dot(RotMatrix,np.transpose(data))) 
        xPCAwhite = np.transpose(xPCAwhite)
        return coe_PC, xPCAwhite
    
    def ZCA_Whitening(self,data):
        Sigma=DimensionReduction.__get_CovarianceMatrix(data)
        RotMatrix, coe_PC, U=np.linalg.svd(Sigma)
        
        epsilon=10**(-5);
        tmp=np.diag(1/np.sqrt(coe_PC+epsilon))
        xZCAwhite = np.dot(RotMatrix,np.dot(tmp,np.dot(RotMatrix,np.transpose(data))))
        xZCAwhite = np.transpose(xZCAwhite)
        return coe_PC, xZCAwhite
    
    
    
    
    def DAFE(self,data,label):
        dim=np.size(data,1)
        TL=(np.unique(label))
        nc=len(TL)
        Sigma_data = np.zeros([dim*nc,dim])
        Mu_data=np.zeros([nc,dim])
        Sw = np.zeros([dim,dim])
        c=0;
        for i in TL:
            c+=1;
            pos=np.where(label==i)
            pos=pos[0]
            tmp=data[pos,:]
            cindex=range((c-1)*dim, c*dim )
            Sigma_data[cindex,:]=np.cov(np.transpose(tmp))
            Sw+=Sigma_data[cindex,:];
            Mu_data[c-1,:]=np.average(tmp,axis=0)             
  
        Sw = Sw / nc;
        Sw = 0.5 * Sw + 0.5 * np.diag(np.diag(Sw)); # regualrization for within-class scatter matrix
        Sb = np.cov(np.transpose(Mu_data))*(nc-1)/nc; # between-calss scatter matrix
        
        C=np.dot(np.linalg.pinv(Sw),Sb)           

        DAFE_vect, DAFE_val, U=np.linalg.svd(C)
        xRot = np.dot(DAFE_vect,np.transpose(data))
        xRot = np.transpose(xRot)   
        return DAFE_vect, xRot
    
    def __get_ZeroMean(data):
        N=np.size(data,0)
        avg=np.average(data)    
        data=data-matlib.repmat(avg, N, 1);
        return data 
    
    def __get_CovarianceMatrix(data):
        N=np.size(data,0)
        x=DimensionReduction.__get_ZeroMean(data)
        Sigma=np.dot(np.transpose(x),x)/N        
        return Sigma               

  
