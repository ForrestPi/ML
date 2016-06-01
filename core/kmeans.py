#coding=utf-8

import numpy as np
import random
def Euclid_dist(x,y):
    if len(y.shape)==1:
        return np.sqrt(np.sum(np.sum((x-y)**2)))
    elif len(y.shape)==2:
        return np.sqrt(np.sum((x-y)**2,axis=1))
    else:
        raise ValueError('error x or y shape')
def dist(x,y):
    '''
    计算两个数据间的距离，使用马氏距离
    '''
    return np.sqrt(np.sum((x-y)**2),axis=1)
def distMat(X,Y):
    '''
    计算两个矩阵间的距里，即矩阵里的每一个数据与另一个矩阵中每一个数据的距离
    '''
    mat=[map(lambda y:dist(x,y),Y) for x in X]
    return np.array(mat)
def sum_dist(data,label,center):
    s=0
    for i in range(data.shape[0]):
        s+=dist(data[i],center[label[i]])
    return s
def kmeans(data,cluster,threshold=1.0e-19,maxIter=100):
    data=np.array(data)
    d1,d2=data.shape
    '''
    find the label
    '''
    batch=np.random.permutation(d1)
    center=data[batch[0:cluster],:]
    print(center.shape)
    labels=np.zeros((d1,))
    last_cost=0
    for ii in xrange(0,d1):
            d=Euclid_dist(data[ii,:],center[labels[ii],:])
            last_cost+=d
    for index in xrange(0,maxIter):
        '''
        寻找每个类的标号
        '''
        for ii in xrange(0,d1):
            this_data=data[ii,:]
            d=Euclid_dist(this_data,center)
            label=np.argmin(d)
            labels[ii]=label
        for ii in xrange(0,cluster):
            batch_no=(labels==ii).nonzero()
            batch=data[batch_no]
            
            m=np.mean(batch,axis=0)
            #print(m.shape)
            center[ii,:]=m
        #print(center)
        current_cost=0
        for ii in xrange(0,d1):
            d=Euclid_dist(data[ii,:],center[labels[ii],:])
            current_cost+=d
        if last_cost-current_cost<threshold:
            break
        else:
            last_cost=current_cost
    return center
'''
def kmeans2(data,cluster,threshold=1.0e-19,maxIter=100):
    m=len(data)
    labels=np.zeros(m)
    #cluster=None
    center=np.array(random.sample(data,cluster))
    s=sum_dist(data,labels,center)
    n=0
    while 1:
        n=n+1
        tmp_mat=distMat(data,center)
        labels=tmp_mat.argmin(axis=1)
        for i in xrange(cluster):
            idx=(labels==i).nonzero()
            m=np.mean(data[idx[0]],axis=0)
            center[i]=m
            #d_i=data[idx[0]]
            #d_i=d_i[0]
        s1=sum_dist(data,labels,center)
        if s-s1<threshold:
            break;
        s=s1
        if n>maxIter:
            break;
    return center
'''
if __name__=='__main__':
    from scipy.io import loadmat,savemat
    data=loadmat(r'E:\code\matlab\DeepLearnToolbox-master\data\mnist_uint8.mat')
    train_x=np.asarray(data['train_x'],np.float)/255.0
    codebook=kmeans(train_x,10)
    savemat('codebook.mat',{'C':codebook})