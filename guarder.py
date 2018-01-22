# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 14:52:49 2017
@author: dawnson
v0.0使用预测得分和预测概率作为特征提取器
"""
'''
@功能：
能够实现对已有异常的良好发掘，并且探测未知异常，将误报率降到最小
@存储优化：
训练阶段：先对所有的构建样本进行类别判断，再按照数组操作，避免了每次都要用分类器判断
实时在线探测：每来一个数据，滑动窗口判断，得到的标签存入数组，避免再次计算
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas.core.frame import DataFrame
'''
参照算法中的队列实现
'''
class Queue(object) :
  def __init__(self, size) :
    self.size = size
    self.queue = []

  def __str__(self) :
    return str(self.queue)

  #获取队列的当前长度
  def getSize(self) :
    return len(self.quene)

  #入队，如果队列满了返回-1或抛出异常，否则将元素插入队列尾
  def enqueue(self, items) :
    if self.isfull() :
      return -1
      #raise Exception("Queue is full")
    self.queue.append(items)

  #出队，如果队列空了返回-1或抛出异常，否则返回队列头元素并将其从队列中移除
  def dequeue(self) :
    if self.isempty() :
      return -1
      #raise Exception("Queue is empty")
    firstElement = self.queue[0]
    self.queue.remove(firstElement)
    return firstElement

  #判断队列满
  def isfull(self) :
    if len(self.queue) == self.size :
      return True
    return False

  #判断队列空
  def isempty(self) :
    if len(self.queue) == 0 :
      return True
    return False

'''
@功能: 通过top训练数据和基类模型，构建出top模型的训练数据
@输入: winsize：窗口大小    data：用于构建top模型的训练数据   model_list：存储两类模型集合的列表  
@返回: 用于构建top模型的数据集
@实现：参见论文中的算法
'''
def build_data(winsize,data,model_list):
    index = 0
    supervise = model_list[0]#model_list为大小为二的列表，分别存储监督模型和非监督模型
    unsupervise = model_list[1]
    dataset = []#用于返回的数据集
    if winsize + index > len(data):
        pass
    else:
        #引入队列，保存每次计算完的结果，避免重复输入到模型中进行计算
        queue = Queue(winsize)
        while index + winsize < len(data):#index的位置就是窗口的最左位置
            while queue.isfull() == False:#当队列不为空，填满队列
                sample = data[index:index+1]#取出样本
                l = []
                for model in supervise:
                    p = model.predict_proba(sample)[0]
                    l.append(p[1])  
                for model in unsupervise:
                    l.append(model.predict_proba(sample)[0])
                queue.enqueue(l)#入队
                index += 1#前移
            #当前队列满，凑够一个窗口的数据,取出数据放入数据集中
            tmp = []
            for i in queue.queue:
                for j in i:
                    tmp.append(j)
            dataset.append(tmp)
            queue.dequeue()
        #print DataFrame(dataset)
        print "the datas for top model are prepared"
        return dataset
    
'''
@功能: 通过top模型的训练数据，构建top模型
@输入: indexlist用于存储训练数据中的索引位置，从而得到数据的标签，形如[100,130,190]表示前100个正常，30个异常，而后60个正常，余下的异常
@返回: 顶级模型
'''
def build_model(dataset,indexlist):
    lable = []
    index = 0
    start = 0
    while index < len(indexlist):
        end = indexlist[index]
        for i in range(start,end):
            if index %2 ==0:
                lable.append(1)
            else:
                lable.append(-1)
        start = end
        index += 1
    for i in range(indexlist[-1],len(dataset)):
        lable.append(-1)
    
    '''
    PLUG IN:
             插入指定的top模型
    '''
    s = DataFrame(dataset)
    rf = RandomForestClassifier(n_estimators=60,min_samples_split = 20, min_samples_leaf = 10,max_depth=11,class_weight='balanced')
    rf.fit(s,lable)
    print "the models builds over"
    return rf


'''
@功能: 通过top模型判断异常,实现为batch处理，实时流处理只需少许更改即可实现
@输入: 与build_data函数输入相同，topmodel为顶级模型
@返回: 实时探测的结果
'''
def predict(topmodel,model_list,winsize,data):
    index = 0
    supervise = model_list[0]
    unsupervise = model_list[1]
    dataset = []
    if winsize + index > len(data):
        pass
    else:
        #引入队列，保存每次计算完的结果，避免重复输入到模型中进行计算
        queue = Queue(winsize)
        while index + winsize < len(data):#index的位置就是窗口的最左位置
            while queue.isfull() == False:#当队列不为空，填满队列
                sample = data[index:index+1]#取出样本
                l = []
                for model in supervise:
                    p = model.predict_proba(sample)[0]
                    l.append(p[1])#可优化以减少计算量  ,下标1指示正常类别
                for model in unsupervise:
                    l.append(model.predict_proba(sample)[0])
                queue.enqueue(l)#入队
                index += 1#前移
            #当前队列满，凑够一个窗口的数据,取出数据放入数据集中
            tmp = []
            for i in queue.queue:
                for j in i:
                    tmp.append(j)
            dataset.append(tmp)
            queue.dequeue()
        s =  DataFrame(dataset)
        print  topmodel.predict(s)
        
        
df = pd.read_csv('smallcovtype.csv')
print "get the data source"
df_normal1 = df[df['10']==1]#提取类别1的数据，被视为正常的数据
df_anamoly1 = df[df['10']==5]#提取类别为5的数据，视为异常类型1的数据
df_anamoly2 = df[df['10']==4]#提取类别为4的数据，视为异常类型2的数据
df_anamoly6 = df[df['10']==6]#提取类别为2的数据，视为异常类型3的数据
print "drop the data label"
df_normal1 = df_normal1.drop(['10'],axis = 1)
df_anamoly1 = df_anamoly1.drop(['10'],axis = 1)
df_anamoly2 = df_anamoly2.drop(['10'],axis = 1)
df_anamoly6 = df_anamoly6.drop(['10'],axis = 1)
print "split train set ,construct set and test set"
normal = df_normal1[:100000]
anamoly = df_anamoly1[:9000]
#这里，我们成训练基类模型的数据为train,训练top模型的数据为construct
train = pd.concat([normal[:9800],anamoly[:200]])#不同异常率实验
construct = pd.concat([normal[20000:20100],anamoly[200:230],normal[20090:20150],df_anamoly6[:40]])#用于构建top模型的数据，很重要！！

train_lable = []
for i in range(0,9800):
    train_lable.append(1)
for i in range(0,200):#训练样本的大小也是一个值得考虑的部分
    train_lable.append(-1)
print "build the base models"

'''
PLUG IN:
可以嵌入任意的异常探测算法
'''
rf = RandomForestClassifier(n_estimators=60,min_samples_split = 20, min_samples_leaf = 10,max_depth=11,class_weight='balanced')
rf.fit(train,train_lable)
gb = GradientBoostingClassifier(random_state=2017,n_estimators= 120,max_depth=10)
gb.fit(train,train_lable)

from sklearn.ensemble import IsolationForest
ilf1 = IsolationForest(n_estimators=100)
ilf2 = IsolationForest(n_estimators=100)
ilf1.fit(train, train_lable)
ilf2.fit(train, train_lable)
#构建参数列表
model_list = []#model_list有两个子列表，子列表中分别是有监督模型和无监督模型
tmp = []
tmp.append(gb)
tmp.append(rf)
model_list.append(tmp)
tmp = []
tmp.append(ilf1)
tmp.append(ilf2)
model_list.append(tmp)
winsize = 6
dataset = build_data(winsize,construct,model_list)
model = build_model(dataset,[100,130,190])#index指示着实际异常的位置

'''
benchmark 
'''
test1 = []#1类异常
test2 = []#2类异常
test6 = []#6类异常
normal_index = 21000
gap_normal = 20
gap1 = 20
gap2 = 30
gap6 = 25
index1 = 100
index2 = 100
index6 = 100
for i in range(0,10):
    tmp = pd.concat([normal[normal_index:normal_index+gap_normal],anamoly[index1:index1+gap1]])
    index1 += gap1
    test1.append(tmp)
    normal_index += gap_normal

for i in range(0,10):
    tmp = pd.concat([normal[normal_index:normal_index+gap_normal],df_anamoly2[index2:index2+gap2]])
    index2 += gap2
    test2.append(tmp)
    normal_index += gap_normal

for i in range(0,10):
    tmp = pd.concat([normal[normal_index:normal_index+gap_normal],df_anamoly6[index6:index6+gap6]])
    index6 += gap6
    test6.append(tmp)
    normal_index += gap_normal
#benchmark结束

for t in test1:
    predict(model,model_list,winsize,t)
for t in test2:
    predict(model,model_list,winsize,t)
for t in test6:
    predict(model,model_list,winsize,t)
