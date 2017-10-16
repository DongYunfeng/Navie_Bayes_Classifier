"""
Created by yfDong on 10/7/17.
"""
import os
import math
import re
import jieba
import numpy as np
import feature_selection

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def translate(bytesstr):
    line = bytesstr.strip().decode('gbk', 'ignore')  # .decode('utf-8', 'ignore')  default Unicode
    p2 = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = "".join(p2.split(line)).strip()
    zh = "".join(zh.split())
    outStr = zh  # 经过相关处理后得到中文的文本
    return outStr


def classifier():
    print('Loading dataset, 80% for training, 20% for testing...')
    dataset_dir_name = "F://Projects/PycharmProjects/Naive Bayes classifier/Chinese documents collection/documents collection"

    movie_reviews = load_files(dataset_dir_name)
    #中文分词
    movie_reviews.data = [" ".join(jieba.cut(translate(doc_str), cut_all=False)) for doc_str in movie_reviews.data]
   #划分测试集和训练集
    doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = \
        train_test_split(movie_reviews.data, movie_reviews.target,
                         test_size=0.2, random_state=0)

    vectorizer = CountVectorizer()
    word_tokenizer = vectorizer.build_tokenizer()
    doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in doc_str_list_train]
    doc_terms_list_test = [word_tokenizer(doc_str) for doc_str in doc_str_list_test]
    # 用互信息提取特征
    #feature_selection.feature_selection_MI(doc_terms_list_train, doc_class_list_train, movie_reviews.target_names,150)

    terms_test_list = feature_selection.get_terms_list(doc_terms_list_test)

    #将提取的特征文件以列表形式存储
    class_features = []

    f=open('F://Projects/PycharmProjects/Naive Bayes classifier/output/feature_selection.txt','r')
    for line in f.readlines():
        temp = []
        for i in line.split(" "):
            if i != "\n":
                temp.append(i)
        class_features.append(temp)

    #朴素贝叶斯
    def bayes(doc_test, class_test):
        word_num = {}
        D = len(doc_terms_list_train)
        len1 = len(doc_test)
        N = len(movie_reviews.target_names)
        y = 0
        b=0
        for clas in doc_class_list_train:#计算训练语料库中类别y包含的文档总数
            if (clas == class_test):
                y += 1
        a = float(math.log(((y + 1) /( D + N)))) #平滑，拉普拉斯修正

        for i in range(0, len1):
            if (doc_test[i] in class_features[class_test]):
                word_num[doc_test[i]] = word_num.get(doc_test[i],0) + 1


        for i in class_features[class_test]:
            b += float(math.log((word_num.get(i,0) + 1)/(y + 12650))) #平滑，拉普拉斯修正
        return a+b

    i = 0
    m = 0

    for doc in doc_terms_list_test:
        tem = 0
        max = bayes(doc, 0)

        for k in range(1,len(movie_reviews.target_names)):
            if(max < bayes(doc,k)):
                max = bayes(doc,k)
                tem = k

        if(doc_class_list_test[i] == tem):
            m += 1
        print(movie_reviews.target_names[doc_class_list_test[i]], movie_reviews.target_names[tem])
        i += 1

    print(float(m/len(doc_terms_list_test)))

classifier()