#!/usr/bin/env python3

import numpy as np
#np.set_printoptions(threshold=np.nan)
import random
import math
from nltk.stem.porter import PorterStemmer
import glob
from itertools import chain

def readEnron(path):
    reader=open(path)
    
    # ignore first line
    reader.readline()

    # second line contains number of emails and dictionary size
    array = reader.readline().split(' ')
    num_of_emails = int(array[0])
    dict_size = int(array[1])
    
    # ignore third line
    reader.readline()
    
    x= np.zeros((num_of_emails,dict_size), dtype=np.int)
    y= np.zeros(num_of_emails, dtype=np.int)
    
    # x[i,j] number of occurences of j-th word in i-th email
    # y[i] i-th email is spam?
    for i in range(num_of_emails):
        array=reader.readline().split(' ')
        int_array=[int(e) for e in array]
        y[i]=int_array[0]
        
        #indexing mind*uck - check encoding.txt file 
        index=0
        for j in range(1,int(len(array)/2)):
            x[i,int_array[2*j-1]-1]=int_array[2*j]
    reader.close()
    return (x,y)

num_outer_loops = 40
lam = 64 #lambda is a reserved word in python

def gaussKernel(a,b):
    Ker=np.zeros((len(a),len(b)))
    for i in range(len(a)):
        for j in range(len(b)): 
            Ker[i,j]=np.exp(-(np.linalg.norm(a[i]-b[j],2))**2/(2*tau*tau))
    return Ker

def learnSVM(x,y):
    #init
    m=len(y)
    x=1*(x>0)
    y=2*y-1
    alpha=np.zeros(m)
    avg_alpha = np.zeros(m)
    #compute kernel
    Ker = gaussKernel(x,x)
    #stochastic gradient descent
    for i in range(num_outer_loops * m):
        #choose a direction
        index = random.randint(0,m-1)
        #compute gradient g
        margin=y[index]*np.dot(Ker[index],alpha)
        g=np.dot(Ker[index],alpha[index])/lam-(margin<1)*y[index]*Ker[index]    
        #apply gradient
        alpha=alpha-g/math.sqrt(i+1)
        avg_alpha+=alpha
        
    avg_alpha=avg_alpha/(num_outer_loops*m)
    
    return avg_alpha

def testSVM(x_test,y_test,avg_alpha,x_train):
    #shorten vectors
    x_test = 1*(x_test>0)
    x_train = 1*(x_train>0)
    y_test=2*y_test-1
    #compute kernel
    Ker = gaussKernel(x_test,x_train)
    #decide
    preds = np.dot(Ker,avg_alpha)
    test_err=np.sum((np.multiply(preds,y_test))<=0)/len(y_test)
    print(np.sum((np.multiply(preds,y_test))<=0))
    print(len(y_test))
    return test_err

tau = 8
num_of_tests = 1
testSizes = ['50', '100', '200', '400', '800', '1400']
#testSizes = ['50','1400']

(m_test, category_test) = readEnron('oldData/enronTrain.2')
for index in range(1):
    err = 0
    for i in range(num_of_tests):
        print("readin")
        (m_train, y_train) = readEnron('oldData/enronTrain.' + str(index+1))
        print("learnin")
        avg_alpha = learnSVM(m_train, y_train)
        print("decidin")
        err += testSVM(m_test, category_test, avg_alpha, m_train)
    print('Train size:', size, 'Error:', err, 'Accurancy:')
