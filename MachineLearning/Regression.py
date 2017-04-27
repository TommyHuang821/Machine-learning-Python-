import numpy as np
class Regression(object):
    def __init__(self):
        self.R=1
        
    class LinearRegression(object):
        def __init__(self):
            self.R=1
        
        def LinearRegression_train(self,TrainingData, response):
            x=TrainingData
            y=response
            Ndata=len(y)
            x=np.c_[np.ones([Ndata,1]),x]
            # Beta_hat=pinv(x'*x)*x'*y; % Least square approach
            xt=x.transpose()
            C=np.dot(xt,x)
            p2=np.dot(xt,y)
            Beta_hat=np.dot(np.linalg.pinv(C),p2)
            
            y_hat=np.dot(x,Beta_hat)
            RMSE=np.sqrt(np.average(np.subtract(y,y_hat)*2))
            self.Beta_hat=Beta_hat 
            self.RMSE=RMSE
            self.y_hat=y_hat
            
        def LinearRegression_test(self, testingData):
            Beta_hat=self.Beta_hat     
            x=testingData
            Ndata=len(x)
            x=np.c_[np.ones([Ndata,1]),x]
            y_test_hat=np.dot(x,Beta_hat)
            return y_test_hat
       
    class RidgeRegression(object):
        def __init__(self):
            self.R=1
            
        def RidgeRegression_train(self,TrainingData, response,lamda):
            x=TrainingData
            y=response
            Ndata=len(x)
            x=np.c_[np.ones([Ndata,1]),x]
            dim=np.size(x,1)
            # Beta_hat=pinv(x'*x+lamda*R)*x'*y; % Least square approach
            R=np.eye(dim)*lamda
            xt=x.transpose()
            C=np.dot(xt,x)+R
            p2=np.dot(xt,y)
            Beta_hat=np.dot(np.linalg.pinv(C),p2)            
            y_hat=np.dot(x,Beta_hat)
            RMSE=np.sqrt(np.average(np.subtract(y,y_hat)*2))
            self.Beta_hat=Beta_hat 
            self.RMSE=RMSE
            self.y_hat=y_hat
        def RidgeRegression_test(self, testingData):
            Beta_hat=self.Beta_hat     
            x=testingData
            Ndata=len(x)
            x=np.c_[np.ones([Ndata,1]),x]
            y_test_hat=np.dot(x,Beta_hat)
            return y_test_hat