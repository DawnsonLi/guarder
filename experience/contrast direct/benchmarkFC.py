# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 14:52:49 2017

@author: dawnson
"""
'''
@功能：
普通算法的benchmark
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas.core.frame import DataFrame
from sklearn.preprocessing import Normalizer

def counter(gap,l):
    c = 0
    for i in range(0,gap):
        if int(l[i]) == -1 :
            c += 1
    return c

def delay(gap,l):
    for i in range(gap,len(l)):
        if int(l[i]) == -1:
            return i-gap
    return -1

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
def calculate(content,factor):
    counter = 0
    for i in content:
        if i == -1:
            counter += 1
    if counter >= factor:
        return -1
    else:
        return 1
def warn(winsize,l,factor):
    q = Queue(winsize)
    i = 0
    ans = []
    while i <len(l):
        while q.isfull() == False:
            q.enqueue(l[i])
        #is full
        content = q.queue
        ans.append( calculate(content, factor) )
        i += 1
        q.dequeue()
    return ans


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
train = pd.concat([normal[:10300],anamoly[:300],df_anamoly6[:100]])#不同异常率实验

#benchmark
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
for i in range(0,10300):
    train_lable.append(1)
for i in range(0,400):#训练样本的大小也是一个值得考虑的部分
    train_lable.append(-1)
print "build the base models"


'''
PLUG IN:
以下填入不同的算法
'''
from sklearn.covariance import EllipticEnvelope
ilf1 = EllipticEnvelope()
ilf1.fit(train,train_lable)

c1 = 0
d1 = 0
winsize = 5
factor = 2
for t in test1:
    l = ilf1.predict(t)
    #l = warn(winsize,l,factor)
    print l[:96]
    print l[96:]
    d = delay(96,l)
    print 'delay:',d#-winsize+1
    d1 += d
    c1 += counter(96,l)

print "**************"
c2 = 0
d2 = 0
for t in test2:
    l = ilf1.predict(t)
    #l = warn(winsize,l,factor)
    print l[:96]
    print l[96:]
    d = delay(96,l)
    print 'delay:',d#-winsize+1
    d2 += d
    c2 += counter(96,l)
print "**************"

c3 = 0
d3 = 0
for t in test6:
    l = ilf1.predict(t)
    #l = warn(winsize,l,factor)
    print l[:96]
    print l[96:]
    d = delay(96,l)
    print 'delay:',d#-winsize+1
    d3 += d
    c3 += counter(96,l)


print 'delay 1:',d1
print 'c1:',c1

print 'delay 2:',d2
print 'c2:',c2

print 'delay 3:',d3
print 'c3:',c3
