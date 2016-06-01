#coding=utf-8

import numpy as np
from cv2 import batchDistance
def sigm(x):
    return 1/(1+np.exp(-x))
def dsigm(x):
    return x*(1-x)
def tanh(x):
    return 1.7159*np.tanh(2.0/3.0*x)
def dtanh(x):
    return 1.7159*2.0/3.0*(1-1.0/(np.power(1.7159,2))*np.power(x,2))
def softmax(x):
    c=3
    tmp=np.exp(c*x)
    n,m=tmp.shape
    s=np.sum(tmp,axis=1)
    n=s.shape[0]
    s=s.reshape((n,1))
    s=np.tile(s,[1,m])
    return tmp/s
class NetLayer(object):
    '''
    神经网络中的层
    '''
    def __init__(self,w,b):
        self.w=w
        self.b=b
        self.vW=np.zeros(w.shape)
        self.vb=np.zeros(b.shape)
        return None
class nnet(object):
    def __init__(self,layers,active_fun='sigm',output_fun='softmax',eta=0.01,momentum=0,show=True):
        if not (type(layers)is list):
            raise ValueError('参数错误')
        n=len(layers)
        self.layer={}
        for ii in xrange(1,n):
            w=(np.random.random_sample((layers[ii-1],layers[ii]))-0.5)*2*4*np.sqrt(6.0/float(layers[ii]+layers[ii-1]))
            #w=np.random.random_sample((layers[ii-1],layers[ii]))*2-1
            b=np.zeros((1,layers[ii]))
            if ii<n-1:
                self.layer[ii]=NetLayer(w,b)
            else:
                self.layer[ii]=NetLayer(w,b)
        self.n=n
        self.eta=eta
        self.out={}
        self.active_fun=active_fun
        self.output_fun=output_fun
        self.momentum=momentum
        return None
    def nnetff(self,data):
        n=self.n
        m=data.shape[0]
        self.out[0]=data
        for ii in xrange(1,n-1):
            a=np.dot(self.out[ii-1],self.layer[ii].w)+np.tile(self.layer[ii].b,[m,1])
            #print(out)
            if self.active_fun=='sigm':
                self.out[ii]=sigm(a)
            elif self.active_fun=='tanh':
                self.out[ii]=tanh(a)
            else:
                raise ValueError('error active_fun %s'%(self.active_fun))
        a=np.dot(self.out[n-2],self.layer[n-1].w)+np.tile(self.layer[n-1].b,[m,1])
        if self.output_fun=='softmax':
            self.out[n-1]=softmax(a)
        elif self.output_fun=='sigm':
            self.out[n-1]=sigm(a)
        elif self.output_fun=='tanh':
            self.out[n-1]=tanh(a)
        return None
    def nnetbp(self,predict_out):
        n=self.n
        d={}
        e=-(predict_out-self.out[n-1])
        m=predict_out.shape[0]
        if self.output_fun=='softmax':
            d[n-1]=e
        elif self.output_fun=='sigm':
            d[n-1]=e*dsigm(self.out[n-1])
        elif self.output_fun=='tanh':
            d[n-1]=e*dtanh(self.out[n-1])
        else:
            raise ValueError('error output_fun %s'%(self.output_fun))
        for ii in xrange(n-2,0,-1):
            if self.active_fun=='sigm':
                d_act=dsigm(self.out[ii])
            elif self.active_fun=='tanh':
                d_act=dtanh(self.out[ii])
            else:
                raise ValueError('error active_fun %s'%(self.active_fun))
            d[ii]=np.dot(d[ii+1],self.layer[ii+1].w.T)*d_act
        for ii in xrange(1,n):
            dw=np.dot(self.out[ii-1].T,d[ii])/m
            dw=self.eta*dw
            db=np.sum(d[ii],axis=0)/m
            db=self.eta*db
            if self.momentum>0:
                self.layer[ii].vW=self.momentum*self.layer[ii].vW+dw
                self.layer[ii].vb=self.momentum*self.layer[ii].vb+db
                dw=self.layer[ii].vW
                db=self.layer[ii].vb
            self.layer[ii].w=self.layer[ii].w-dw
            self.layer[ii].b=self.layer[ii].b-db
        return None
    def nnetff_fast(self,data):
        out=data
        m=data.shape[0]
        n=self.n
        for ii in xrange(1,n-1):
            a=np.dot(out,self.layer[ii].w)+np.tile(self.layer[ii].b,[m,1])
            if self.active_fun=='sigm':
                out=sigm(a)
            elif self.active_fun=='tanh':
                out=tanh(a)
            else:
                raise ValueError('error active_fun %s'%(self.active_fun))
        a=np.dot(out,self.layer[n-1].w)
        if self.output_fun=='softmax':
            out=softmax(a)
        elif self.output_fun=='sigm':
            out=sigm(a)
        elif self.output_fun=='tanh':
            out=tanh(a)
        else:
            raise ValueError('error output_fun %s'%(self.output_fun))
        return out
    def predict(self,data):
        out=self.nnetff_fast(data)
        return np.argmax(out,axis=1)
    def train(self,train_x,train_y,epoch=20,batch_size=1):
        #train_x=np.array(train_x,dtype=np.float32)
        #train_y=np.array(train_y,dtype=np.float32)
        n=train_x.shape[0]
        assert n%batch_size==0
        n_batch=n/batch_size
        for ii in xrange(0,epoch):
            batch=np.random.permutation(n)
            for jj in xrange(0,n_batch):
                
                t_batch=train_x[batch[jj*batch_size:(jj+1)*batch_size]]
                t_y=train_y[batch[jj*batch_size:(jj+1)*batch_size]]
                self.nnetff(t_batch)
                self.nnetbp(t_y)
    def test(self,test_x,test_y):
        n=float(test_y.shape[0])
        predict_label=self.predict(test_x)
        label=np.argmax(test_y,axis=1)
        return np.sum(np.array(predict_label==label,dtype=np.int))/n
    def train_one_epoch(self,train_x,train_y):
        self.nnetff(train_x)
        self.nnetbp(train_y)
if __name__=='__main__':
    from scipy.io import loadmat
    data=loadmat(r'E:\code\matlab\DeepLearnToolbox-master\data\mnist_uint8.mat')
    train_x=np.asarray(data['train_x'],np.float)/255.0
    train_y=np.asarray(data['train_y'],np.float)
    test_x=np.asarray(data['test_x'],np.float)/255.0
    test_y=np.asarray(data['test_y'],np.float)
    del data
    nn=nnet([784,100,100,10],active_fun='sigm',output_fun='softmax',eta=0.1,momentum=0.5)
    nn.train(train_x,train_y,epoch=10,batch_size=100)
    accuray=nn.test(test_x,test_y)
    print('Last accuray=%f%%'%(accuray*100))
    
    
            
            