#!/usr/bin/env python
#-*- coding:utf-8 -*-

import logRegres
from numpy import *
dataArr,labelMat=logRegres.loadDataSet()
#weights = logRegres.gradAscent(dataArr,labelMat)
#logRegres.plotBestFit(weights.getA())
weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
logRegres.plotBestFit(weights)

logRegres.multiTest()
    
