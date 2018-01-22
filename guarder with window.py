# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 14:52:49 2017

@author: dawnson
使用启发式规则对结果进行过滤，是实验的一项内容
"""
'''
@功能：
能够实现对已有异常的良好发掘，并且探测未知异常，将噪声降到最小
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
from sklearn.preprocessing import Normalizer

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
统计误报总数,gap为异常点k
'''
def counter(gap,l):
    c = 0
    for i in range(0,gap):
        if int(l[i]) == -1 :
            c += 1
    return c

'''
统计延迟delay
'''
def delay(gap,l):
    for i in range(gap,len(l)):
        if int(l[i]) == -1:
            return i-gap
    return -1

'''
统计窗口中的异常率是否达到阈值factor
'''
def calculate(content,factor):
    counter = 0
    for i in content:
        if i == -1:
            counter += 1
    if counter >= factor:
        return -1
    else:
        return 1
    
'''
实际的报警逻辑，实现异常过滤，避免过多误报
'''
def warn(winsize,l,factor):
    q = Queue(winsize)
    i = 0
    ans = []
    while i <len(l):
        while q.isfull() == False:
            q.enqueue(l[i])
        #is full
        content = q.queue#取出当前窗口中的数据
        ans.append( calculate(content, factor) )
        i += 1
        q.dequeue()
    return ans



'''
@功能: 通过top训练数据和基类模型，构建出top模型的训练数据
@输入: winsize：窗口大小    data：用于构建top的训练数据  model_list：存储两类模型集合的列表  
@返回: 用于构建top模型的数据集
'''
def build_data(winsize,data,model_list):
    index = 0
    supervise = model_list[0]
    unsupervise = model_list[1]
    triple_list = []
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
                    l.append(p[1])#可优化以减少计算量  
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
        print DataFrame(dataset)
        return dataset
'''
@功能: 通过top模型的训练数据，构建top模型
@输入: indexlist用于存储训练数据中的索引位置，形如[100,130,190]表示前100个正常，30个异常，而后60个正常，余下的异常
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
   
    s = DataFrame(dataset)
    normalizer = Normalizer().fit(s)  # fit does nothing
    s = normalizer.fit_transform(s)
    print s
    rf = RandomForestClassifier(n_estimators=60,min_samples_split = 20, min_samples_leaf = 10,max_depth=11,class_weight='balanced')
    rf.fit(s,lable)
    print "the models builds over"
    return rf,normalizer

'''
@功能: 通过top模型判断异常
@输入: 与build_data函数输入相同，topmodel为顶级模型
@返回: 实时探测的结果
'''
def predict(topmodel,normalizer,model_list,winsize,data):
    index = 0
    supervise = model_list[0]
    unsupervise = model_list[1]
    triple_list = []
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
        s = normalizer.fit_transform(s)
        return  topmodel.predict(s)
        
from sklearn.utils import shuffle  

df = pd.read_csv('smallcovtype.csv')
print "get the data source"
df_normal1 = df[df['10']==1]#提取类别1的数据，被视为正常的数据
df_anamoly1 = df[df['10']==5]#提取类别为5的数据，视为异常类型1的数据
df_anamoly2 = df[df['10']==4]#提取类别为4的数据，视为异常类型2的数据
df_anamoly6 = df[df['10']==6]#提取类别为2的数据，视为异常类型3的数据
df_normal1 = shuffle(df_normal1)
df_anamoly1 = shuffle(df_anamoly1)
df_anamoly2 = shuffle(df_anamoly2)
df_anamoly6 = shuffle(df_anamoly6)
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
construct = pd.concat([normal[9800:10100],anamoly[200:300],normal[10100:10300],df_anamoly6[:100]])#用于构建top模型的数据，很重要！！
'''
Benchmark
'''
test1 = []#1类异常
test2 = []#2类异常
test6 = []#6类异常
normal_index = 21000
gap_normal = 100
gap1 = 30
gap2 = 30
gap6 = 30
index1 = 300
index2 = 0
index6 = 100
for i in range(0,30):
    tmp = pd.concat([normal[normal_index:normal_index+gap_normal],anamoly[index1:index1+gap1]])
    index1 += gap1
    test1.append(tmp)
    normal_index += gap_normal

for i in range(0,30):
    tmp = pd.concat([normal[normal_index:normal_index+gap_normal],df_anamoly2[index2:index2+gap2]])
    index2 += gap2
    test2.append(tmp)
    normal_index += gap_normal

for i in range(0,30):
    tmp = pd.concat([normal[normal_index:normal_index+gap_normal],df_anamoly6[index6:index6+gap6]])
    index6 += gap6
    test6.append(tmp)
    normal_index += gap_normal
#benchmark结束

train_lable = []
for i in range(0,9800):
    train_lable.append(1)
for i in range(0,200):#训练样本的大小也是一个值得考虑的部分
    train_lable.append(-1)
print "build the base models"

'''
PLUG IN
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

winsize = 5
dataset = build_data(winsize,construct,model_list)
model,normalizer= build_model(dataset,[300,400,600])#index指示着实际异常的位置

'''
c = 0
for t in test1:
    l = predict(model,normalizer,model_list,winsize,t)
    print l[:96]
    print l[96:]
    print 'delay:',delay(96,l)#-winsize+1
    c += counter(96,l)
print c
print "**************"
c = 0
for t in test2:
    l = predict(model,normalizer,model_list,winsize,t)
    print l[:96]
    print l[96:]
    print 'delay:',delay(96,l)
    c += counter(96,l)
print "**************"
print c
c = 0
for t in test6:
    l = predict(model,normalizer,model_list,winsize,t)
    print l[:96]
    print l[96:]
    print 'delay:',delay(96,l)
    c += counter(96,l)
print c

'''
f = open('log.txt','w')
c1 = 0
d1 = 0
winsize = 5
factor = 2
for t in test1:
    l = predict(model,normalizer,model_list,winsize,t)
    l = warn(winsize,l,factor)
    print l[:96]
    print l[96:]
    f.write(str(l[:96]))
    f.write(str(l[96:]))
    d = delay(96,l)
    print 'delay:',d#-winsize+1
    d1 += d
    c1 += counter(96,l)
print c1
f.write('************************')
print "**************"
c2 = 0
d2 = 0
for t in test2:
    l = predict(model,normalizer,model_list,winsize,t)
    l = warn(winsize,l,factor)
    print l[:96]
    print l[96:]
    f.write(str(l[:96]))
    f.write(str(l[96:]))
    d = delay(96,l)
    print 'delay:',d#-winsize+1
    d2 += d
    c2 += counter(96,l)
print "**************"
f.write('************************')
print c2
c3 = 0
d3 = 0
for t in test6:
    l = predict(model,normalizer,model_list,winsize,t)
    l = warn(winsize,l,factor)
    print l[:96]
    print l[96:]
    f.write(str(l[:96]))
    f.write(str(l[96:]))
    d = delay(96,l)
    print 'delay:',d#-winsize+1
    d3 += d
    c3 += counter(96,l)
print c3
f.close()
print 'delay 1:',d1
print 'c1:',c1

print 'delay 2:',d2
print 'c2:',c2

print 'delay 3:',d3
print 'c3:',c3
