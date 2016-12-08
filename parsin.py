#!/usr/bin/env python3
import numpy as np
#np.set_printoptions(threshold=np.nan)
import random
import math
from nltk.stem.porter import PorterStemmer
import glob
from itertools import chain

def stemStopWords(filename):
    porter_stemmer = PorterStemmer()
    with open(filename, "r",encoding='utf-8', errors='ignore') as f:
         return set([porter_stemmer.stem(line.rstrip('\n')) for line in f])

#load stemmed stopWords
stopWords = stemStopWords("oldData/stopwords.txt")

def parseEmail(filename):
    porter_stemmer = PorterStemmer()
    with open(filename, "r",encoding='utf-8', errors='ignore') as f:
        mail = sorted([porter_stemmer.stem(word) for line in f for word in line.rstrip('\n').split(' ')])
    c = 0
    parsed=[]
    prev = mail[0]
    for word in mail:
        if word==prev:
            c+=1
        else:
            parsed.append((prev,c))
            c=1
        prev = word
    parsed.append((prev,c)) 
    return [(prev,c) for prev,c in parsed if prev not in stopWords]

def emailToNum(wordList,spam,filename):
    parsed = parseEmail(filename)
    wList = wordList + [x for x,c in parsed if x not in set(wordList)]
    a = 0
    if spam:
        a = 1
    return wList,[a]+list(chain.from_iterable([(wList.index(x)+1,c) for x,c in parsed]))+[-1]

def enronToNew(num):
    mailCount = 0
    wordList = []
    emailList = []
    for ham in glob.glob("./oldData/enron{0}/ham/*".format(num)):
        print(ham, mailCount)
        mailCount+=1
        wordList, numList = emailToNum(wordList,False,ham)
        emailList.append(numList)
    for spam in glob.glob("./oldData/enron{0}/spam/*".format(num)):
        print(spam, mailCount)
        mailCount+=1
        wordList, numList = emailToNum(wordList,True,spam)
        emailList.append(numList)
    #return #emails, #different words, dictionary
    return mailCount,len(wordList),wordList,emailList

def printEnron(num):
    mailcount, wlen, wList, eList = enronToNew(num)
    f = open("oldData/enronTrain.{0}".format(num),"w")
    f.write("enron{0}\n".format(num))
    f.write(str(mailcount)+" "+str(wlen)+"\n")
    f.write(" ".join(wList)+"\n")
    for email in eList:
        f.write(" ".join(str(w) for w in email)+'\n')
    f.close()

#need only 5 and 6
for i in range(5,6):
    printEnron(i+1)
    print(i)
