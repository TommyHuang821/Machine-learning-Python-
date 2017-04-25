import numpy as np
import matplotlib.pyplot as plt
from Classifier import GaussianClassifier
mean=[0,1]
cov=[[0.1,0.1],[0.2,0.3]]
d1=np.random.multivariate_normal(mean, cov, 100)

mean=[-1,-1]
cov=[[0.1,0.1],[0.2,0.3]]
d2=np.random.multivariate_normal(mean, cov, 100)
data=np.concatenate((d1,d2),axis=0)
del d1, d2            
               
label=(np.ones([200]))
label[100:201]=2

 

## Gaussian Classifier
GC=GaussianClassifier.GaussianClassifier(data,label,data)
GC.GaussianClassifier_train()
p=GC.GaussianClassifier_test()



plt.scatter(data[label==1,0],data[label==1,1],color="b")
plt.hold('on')
plt.scatter(data[label==2,0],data[label==2,1],color="r")
PL = np.argmax(p,axis=1)+1
plt.scatter(data[PL==1,0],data[PL==1,1],color="b", marker='+')
plt.scatter(data[PL==2,0],data[PL==2,1],color="r", marker='+')
plt.show()

