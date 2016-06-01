#coding=utf-8
import numpy as np
'''
参数1:特征值向量，  
参数2:比率  
返回值：k(符合指定比率的topK k值)  
'''
def setK(eigVals,rate = 0.9):  
    eigValInd = np.argsort(eigVals)  #对特征值进行排序  
    for i in range(1,eigVals.size+1):  
        topK = eigValInd[:-(i + 1):-1]  
        eigVal = eigVals[:, topK]  
        a = eigVal.sum()  
        b = eigVals.sum()  
        print a/b  
        if a/b >= rate:  
            break;  
    return i  
#功能：给定一个矩阵，返回 经PCA算法降过维的 矩阵（降维的程度由rate决定）  
#     如果要指定k,可直接修改这条语句“ k = setK(eigVals,rate)”  
#参数1:矩阵  
#参数2:要取特征值的比率（rate = sum（topK特征值）/sum(特征值总和)）  
#返回值：经PCA算法降过维的 矩阵  

def pca(dataMat,rate=0.9):
    meanVals=np.mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals  
    stded=meanRemoved/np.std(dataMat) #用标准差归一化
    covMat=np.cov(stded,rowvar=0)     #求协方差矩阵
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    k = setK(eigVals,rate)   #get the topNfeat  
    eigValInd = np.argsort(eigVals)  #对特征值进行排序  
    eigValInd = eigValInd[:-(k + 1):-1]  #get topNfeat  
    redEigVects = eigVects[:, eigValInd]       # 除去不需要的特征向量  
    lowDDataMat = stded * redEigVects    #求新的数据矩阵  
    #reconMat = (lowDDataMat * redEigVects.T) * std(dataMat) + meanVals  #对矩阵还原  
    return lowDDataMat  
