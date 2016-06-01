#!/usr/bin/env python
#-*- coding:utf-8 -*-
import bayes
# listOPosts,listClasses = bayes.loadDataSet()
# myVocabList = bayes.createVocabList(listOPosts)
# print myVocabList
# print bayes.setOfWords2Vec(myVocabList,listOPosts[0])
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
# p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)
# print pAb
# print sum(p0V)
# 
# bayes.testingNB()


# import re
# regEx=re.compile('\\W*')
# emailText=open('email/ham/6.txt').read()
# listOfTokens=regEx.split(emailText)
# print listOfTokens

#bayes.spamTest()
import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList,pSF,pNY=bayes.localWords(ny,sf)
bayes.getTopWords(ny,sf)