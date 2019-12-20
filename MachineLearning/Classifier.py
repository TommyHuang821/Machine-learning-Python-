import numpy as np
from numpy import matlib
class Classifier(object):
    def __init__(self): pass
    

    class QDC(object): # quadratic discriminant classifier 
        def __init__(self): pass
            
        def QDC_train(self,data,label):
            mu=[]
            sigma=[]
            labelindex=np.unique(label)
            for item in labelindex:
                pos=np.where(label==item)
                p1=pos[0]
                mu.append(np.average(data[p1], axis=0))
                sigma.append(np.cov(data[p1].T))
            self.mu=mu
            self.sigma=sigma
            self.labelindex=labelindex
        
        def QDC_test(self,testdata):
            mu=self.mu
            sigma=self.sigma
            labelindex=self.labelindex
            Ny,dim=testdata.shape
            '''When yor run a larger number of testing data, the out-of-memory error may occur.
                Hence, we have to do a partial segmentation for the testing data.
                In my code, if the number of testing data is larger than 10,000, 
                   the mechanism of partial segmentation would work. 
            '''
            if Ny>10000: 
                P=(np.zeros([Ny,len(labelindex)])) 
                epoch=int(np.floor(Ny/10000))
                for epo in range(epoch):
                    index=range(0+epo*10000,(1+epo)*10000)
                    tmptestdata=testdata[index,:]
                    pp=Classifier.QDC.__QDC_test_calculatProbability(tmptestdata,mu,sigma,labelindex) 
                    P[index,:]=pp
                index=range(index[-1],Ny)
                tmptestdata=testdata[index,:]
                pp=Classifier.QDC.__QDC_test_calculatProbability(tmptestdata,mu,sigma,labelindex) 
                P[index,:]=pp                
            else:
                P=Classifier.QDC.__QDC_test_calculatProbability(testdata,mu,sigma,labelindex)    
            return P
        
        def __QDC_test_calculatProbability(testdata,mu,sigma,labelindex):
            Ny,dim=testdata.shape
            P=(np.zeros([len(labelindex),Ny]))
            for i in range(len(labelindex)):
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
            p1=matlib.repmat(np.sum(P, axis=0),len(labelindex),1)
            P=P/p1       
            P=np.transpose(P)
            return P
        
    class NBC(object): # Naive Bayes Classifier 
        def __init__(self): pass
            
        def NBC_train(self,data,label):
            
            N,dim=data.shape
            labelindex=np.unique(label);
                                
            Mu=np.zeros([len(labelindex),dim])
            Sigma=np.zeros([len(labelindex),dim])
            invSigma=np.zeros([len(labelindex),dim])
            gaussiancoeff=np.zeros([len(labelindex),dim])
            
            priori=np.zeros([len(labelindex),1])
            count=0
            for item in labelindex:
                pos=np.where(label==item)
                priori[count]=len(pos)/N
                p1=pos[0]
                for di in range(dim):
                    mu=(np.average(data[p1,di], axis=0))
                    sigma=(np.cov(data[p1,di].T))
                    Mu[count,di]=mu
                    Sigma[count,di]=sigma
                    invSigma[count,di]=sigma**(-1)
                    gaussiancoeff[count,di]=-0.5*(dim*np.log(2*np.pi)+np.log(sigma))
                count+=1   
            self.Mu=Mu
            self.Sigma=Sigma
            self.invSigma=invSigma
            self.gaussiancoeff=gaussiancoeff
            self.priori=priori
            self.labelindex=labelindex

            
        def NBC_test(self,testdata):
            labelindex=self.labelindex
            priori=self.priori
            Mu=self.Mu
            invSigma=self.invSigma
            gaussiancoeff=self.gaussiancoeff
            N,dim=testdata.shape
            logLikelihood=np.zeros([N,len(labelindex)])
            Posterprobability=np.zeros([N,len(labelindex)])
            count=0
            for item in labelindex:
                tmpPr=priori[count];    
                tmplogLikelihood=np.zeros([N,1]);
                for di in range(dim):
                    mu=Mu[count,di]            
                    invS=invSigma[count,di]            
                    gf=gaussiancoeff[count,di]            
                    zmdata=testdata[:,di]-mu
                    tmpm=-0.5*zmdata*(invS*zmdata)
                    tmplogLikelihood[:,0]=tmplogLikelihood[:,0]+(tmpPr+tmpm+gf)
                logLikelihood[:,count]=tmplogLikelihood[:,0]
                count+=1
                       
            Posterprobability=np.exp(logLikelihood)/np.transpose(np.matlib.repmat(np.transpose(np.sum(np.exp(logLikelihood),axis=1)),len(labelindex),1))
            return Posterprobability

    class NearestNeighbor(object):
        def __init__(self,**kwargs): 
            for name, value in kwargs.items():
                if name == 'k': self.k=value
            try:    self.k
            except: 
                self.k=3
                print('kNN: k=3 by default')
            
            
        def kNN(self,traindata,label_train,testdata):  
            k=self.k
            
            Nx,dimx=np.shape(traindata)
            Ny,dimy=np.shape(testdata)
            
            LabelIndex = np.unique(label_train)
            nc = len(LabelIndex)
            vote = np.zeros((Ny,nc))
            
            ## Count the distance between traindata and testdata ##
            assemble=np.concatenate((testdata,traindata))
            
            temp=np.array(np.zeros((Nx+Ny,1)))
            temp[:,0]=np.sum(assemble**2,axis=1)
            
            tmp1=np.kron(np.ones((1,Nx+Ny)),temp)
            tmp2=np.kron(np.ones((Nx+Ny,1)),temp.transpose())
            tmp12=np.dot(assemble,assemble.transpose())
            dist=tmp1+tmp2-2*tmp12
            d = dist[0:Ny,Ny:Ny+Nx]
            sort_ind=d.argsort(axis=1)
            id_temp=sort_ind.flatten()
            
            pre_label=label_train[id_temp]
            
            kNNlabel=np.reshape(pre_label,(Ny,Nx))
            #### vote ###
            Decisionlabel=np.zeros(Ny)
            for i in range(Ny):
                for j in range(k):
                    vote[i,int(kNNlabel[i,j])] +=  1;
                Decisionlabel[i]=LabelIndex[np.where(vote[i,:]==max(vote[i,:]))[0]]
            return Decisionlabel
                    
