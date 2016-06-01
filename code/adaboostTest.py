#!/usr/bin/env python
#-*- coding:utf-8 -*-
import adaboost
from numpy import *
#dataMat,classLabels=adaboost.loadSimData()

# D=mat(ones((5,1))/5)
# print D
# 
# bestStump,minError,bestClasEst=adaboost.buildStump(dataMat,classLabels,D)
# print bestStump

# classifierArray=adaboost.adaBoostTrainDS(dataMat,classLabels,30)
# print adaboost.adaClassify([[5,5],[0,0]],classifierArray)

dataArr,labelArr=adaboost.loadDataSet('./dataSet/horseColicTraining2.txt')
classifierArray,aggClassEst=adaboost.adaBoostTrainDS(dataArr,labelArr,10)
adaboost.plotROC(aggClassEst.T,labelArr)
testArr,testLabelArr=adaboost.loadDataSet('./dataSet/horseColicTest2.txt')
prediction10=adaboost.adaClassify(testArr,classifierArray)
errArr=mat(ones((67,1)))
print errArr[prediction10!=mat(testLabelArr).T].sum()
