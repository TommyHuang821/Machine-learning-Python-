import numpy as np # import numpy lib (矩陣運算用)
import matplotlib.pyplot as plt # (畫圖用)
# MachineLearning 我自己寫的lib
from MachineLearning import GaussianClassifier # 高斯分類器其實就是QDC
from MachineLearning import DimensionReduction # 降維度
'''
 產生兩組2維資料，各100個點
 第一組群心[0,1]
 coveriance matrix ([0.1 0.1]
                    [0.2 0.3])
  第一組群心[3,3]
 coveriance matrix ([0.1 0.1]
                    [0.2 0.3])
'''
mean=[0,1]
cov=[[0.1,0.1],[0.1,0.3]]
d1=np.random.multivariate_normal(mean, cov, 100)

mean=[2,2]
cov=[[0.1,0.1],[0.1,0.3]]
d2=np.random.multivariate_normal(mean, cov, 100)
data=np.concatenate((d1,d2),axis=0) # 把資料合起來
del d1, d2   #刪掉不要的變數         
               
# 將兩組資料給label，第一組是1，第二組是2
label=(np.ones([200]))
label[100:201]=2
     
     
'''    
這邊做降維，因為是demo，所以不會將投影的資料拿去分類用
'''
# 宣告一個降維的原件叫DR
DR=DimensionReduction.DimensionReduction()
# PCA
coe_PC, xdata=DR.PCA(data)     
# LCA/DAFE
coe_DAFE, xdata_DAFE=DR.DAFE(data,label)



'''
這邊是分類器，目前只寫了Gaussian Classifier(QDC)
這邊處理方式和降維不太一樣，因為我目前只寫一個分類器，如果有多個以後會改成跟降維一樣
'''
# 宣告一個GaussianClassifier的原件叫GC,且要將資料塞進去
GC=GaussianClassifier.GaussianClassifier(data,label,data)
# training GaussianClassifier
GC.GaussianClassifier_train()
# testing GaussianClassifier, 輸出是機率
p=GC.GaussianClassifier_test()
PL = np.argmax(p,axis=1)+1 # PL是分類器預測的類別

# 畫圖
plt.figure(1) # 開一張圖
plt.subplot(221) # 這個寫法等於MATLAB的subplot(2,2,1)
plt.scatter(data[label==1,0],data[label==1,1],color="b",marker=r'$\bigodot$')
plt.scatter(data[label==2,0],data[label==2,1],color="r",marker=r'$\bigodot$')
plt.scatter(data[PL==1,0],data[PL==1,1],color="b", marker='+')
plt.scatter(data[PL==2,0],data[PL==2,1],color="r", marker='+')
plt.title('Original data, + is classification result')
plt.subplot(222)
plt.scatter(xdata[label==1,0],xdata[label==1,1],color="b",marker=r'$\bigodot$')
plt.scatter(xdata[label==2,0],xdata[label==2,1],color="r",marker=r'$\bigodot$')
plt.title('PCA Projection')
plt.subplot(223)
plt.scatter(xdata_DAFE[label==1,0],xdata_DAFE[label==1,1],color="b",marker=r'$\bigodot$')
plt.scatter(xdata_DAFE[label==2,0],xdata_DAFE[label==2,1],color="r",marker=r'$\bigodot$')
plt.title('LDA Projection')
plt.show()

