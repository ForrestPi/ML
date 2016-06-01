#coding=utf-8

from rbm import rbm
from nnet import nnet
import numpy as np
#dbn(深度信任网络)
class dbn(object):
    #初始化网络
    #参数：
    #layers网络中每层的神经元个数（包括输入层）
    #momentum 动量
    #alpha alpha
    #show 是否显示运行状态
    #输出：
    #None
    def __init__(self,layers,momentum=0,alpha=1,show=True):
        self.lrbm={}
        self.size=layers
        for ii in xrange(1,len(layers)):
            self.lrbm[ii]=rbm(n_in=layers[ii-1],n_out=layers[ii],momentum=momentum,alpha=alpha,show=show)
        self.n=len(layers)
        return None
    #训练dbn
    #参数：
    #data训练数据
    #epoch训练次数
    #batch_size 分组大小
    #输出：
    #None
    def train(self,data,epoch,batch_size):
        out=data
        for ii in xrange(1,self.n):
            self.lrbm[ii].train(out,epoch,batch_size)
            out=self.lrbm[ii].rbmff(out)
        return None
    #将dbn转化为nnet
    #参数：
    #n_out 输出层个数
    #输出：
    #nnet
    def dbn2nnet(self,n_out):
        if type(n_out) is list:
            size=self.size+n_out
        else:
            n_out=[n_out]
            size=self.size+n_out
        ann=nnet(layers=size,active_fun='sigm',output_fun='softmax',momentum=0,show=True)
        for ii in xrange(1,ann.n-1):
            ann.layer[ii].w=self.lrbm[ii].w.T
            ann.layer[ii].b=self.lrbm[ii].c.T
        for ii in xrange(1,ann.n):
            print('w shape=%s',ann.layer[ii].w.shape)
            print('b shape=%s',ann.layer[ii].b.shape)
        return ann
if __name__=='__main__':
    from scipy.io import loadmat
    data=loadmat(r'E:\code\matlab\DeepLearnToolbox-master\data\mnist_uint8.mat')
    train_x=np.asarray(data['train_x'],np.float)/255.0
    train_y=np.asarray(data['train_y'],np.float)
    test_x=np.asarray(data['test_x'],np.float)/255.0
    test_y=np.asarray(data['test_y'],np.float)
    nn=dbn([784,100,100])
    nn.train(train_x,epoch=1,batch_size=50)
    ann=nn.dbn2nnet(10)
    ann.train(train_x, train_y,epoch=20, batch_size=50)
    accuray=ann.test(test_x, test_y)
    print('Last accuray=%f%%'%(accuray*100))