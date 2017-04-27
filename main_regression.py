import numpy as np # import numpy lib (矩陣運算用)
import matplotlib.pyplot as plt # (畫圖用)
# MachineLearning 我自己寫的lib
from MachineLearning import Regression as Reg

ALL_data=np.genfromtxt("RegressionExample.txt", delimiter=' ',dtype =float)

x=ALL_data[:,1:4]
y=ALL_data[:,0]
      
      
RG=Reg.Regression.LinearRegression()
RG.LinearRegression_train(x,y)
y_hat1=RG.LinearRegression_test(x)

RRG=Reg.Regression.RidgeRegression()
RRG.RidgeRegression_train(x,y,0.5)
y_hat2=RRG.RidgeRegression_test(x)
      

plt.plot(y,y,color="k", label='Ground Truth')
plt.scatter(y,y_hat1,color="b", marker='x', label='Linear Regression')
plt.scatter(y,y_hat2,color="r", marker=r'$\bigodot$', label='Ridge Regression')
plt.axis('scaled')
plt.legend(loc='lower right',numpoints=1, handlelength=1, handletextpad=1, labelspacing=1,
        ncol=1,mode="None",borderpad=1, fancybox=True)
plt.show()
