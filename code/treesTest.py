#!/usr/bin/env python
#-*- coding:utf-8 -*-
import trees
#part one 
# myDat,labels = trees.createDataSet()
# print myDat
# print trees.calcShannonEnt(myDat)
# 
# print trees.splitDataSet(myDat,0,1)
# print trees.splitDataSet(myDat,1,1)
# myTree = trees.createTree(myDat,labels)
# print myTree
# trees.storeTree(myTree,'classifierStorage.txt')
# print trees.grabTree('classifierStorage.txt')

#part two
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print lensesTree
import treePlotter
treePlotter.createPlot(lensesTree)
