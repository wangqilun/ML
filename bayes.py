#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                    ['maybe','not','take','him','to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','steak','how','to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec


def createVocabList(dataSet):
    """创建词汇表，包含所有文档中出现的不重复的词表"""
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """词集模型   输出文档向量，向量的每一元素为1或0，分别表示词汇表的单词在输入文档中是否出现"""
    returnVec =  [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word： %s is not in my Vocabulary!" % word
    return returnVec


def bagOfWords2VecMN(vocabList,inputSet):
    """词袋模型   输出文档向量，向量的每一元素n，表示词汇表的单词在输入文档中出现的次数"""
    returnVec =  [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """计算概率(类条件概率 和 类概率)"""
    numTrainDocs = len(trainMatrix)  # 矩阵行数
    numWords = len(trainMatrix[0])   # 矩阵列数
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 文档属于侮傉性文档的概率
    p0Num = ones(numWords)          # 初始化分子 分母
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]       # 向量相加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)     # 对每个元素做除法
    p0Vect = log(p0Num / p0Denom)     # 使用log函数，避免下溢出
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Clasify, p0Vec, p1Vec, pClass1):
    """朴素贝叶斯分类器"""
    p1 = sum(vec2Clasify * p1Vec) + log(pClass1)   # p(w|ci)p(ci) 取对数
    p0 = sum(vec2Clasify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:   # 比较类别的概率，返回大概率对应的类别标签
        return 1
    else:
        return 0


def testingNB():
    """便利函数，该函数封装所有的操作"""
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trianMat = []
    for postinDoc in listOPosts:
        trianMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trianMat), array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb)


def textParse(bigString):
    """文本解析"""
    import re
    listOfTokens = re.split(r'\w*', bigString)  # 正则划分
    # 去掉少于两个字符的字符串，并将所有的字符串转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def spamTest():
    """垃圾邮件测试"""
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):  # 一共50封邮件
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):  # 随机选取十封邮件为测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  # 删除指定值得元素 li = [1,2,4,5]  li.remove(4)  # li = [1,2,5]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet: 
        # 训练集中的每封邮件基于词汇表并使用 setOfWords2Vec()函数来构建词向量
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet: # 遍历测试集，对其中的每封邮件进行分类
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1   # 如果邮件分类错误，则错误数加1
    print 'the error rate is:', float(errorCount)/len(testSet)
