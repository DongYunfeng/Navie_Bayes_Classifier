"""
Created by yfDong on 10/7/17.
"""
import math
def get_terms_list(doc_terms_list):
    terms_list=[]
    for doc in doc_terms_list:
        for term in doc:
            if(term not in terms_list):
                terms_list.append(term)
    return terms_list


def feature_selection_MI(doc_terms_list, doc_class_list,target_names,split_num):
    f = open('F://Projects/PycharmProjects/Naive Bayes classifier/output/feature_selection.txt', 'w')
    terms_list = get_terms_list(doc_terms_list)
    target_num = len(target_names)
    N = len(doc_terms_list)
    for clas in range(0,target_num):
        mi = []
        mi_dict = {}
        for term in terms_list:
            N11, N10, N01, N00 = 0,0,0,0
            for i in range(0,N):
                if(term in doc_terms_list[i] and clas == doc_class_list[i]):
                    N11+=1
                elif(term in doc_terms_list[i] and clas != doc_class_list[i]):
                    N10+=1
                elif(term not in doc_terms_list[i] and clas == doc_class_list[i]):
                    N01+=1
                else:
                    N00+=1

            N1_=N10+N11
            N_1=N01+N11
            N0_=N00+N01
            N_0=N10+N00
            if(N11>0 and N00>0 and N01>0 and N10>0):
                mi.append(float(N11/N*math.log(N*N11/(N1_*N1_),2)+N01/N*math.log(N*N01/(N0_*N_1),2)+
                      N10/N*math.log(N*N11/(N1_*N_0),2)+N00/N*math.log(N*N00/(N0_*N_0),2)))
            else:
                mi.append(0)
        #print(mi)
        for i in range(0,len(mi)):
            mi_dict[i] =mi[i]

        j=0
        word_list = []
        #以互信息降序选出前n个相关特征词
        for index, value in sorted(mi_dict.items(), key=lambda item: item[1], reverse=True):
            if (j < split_num):
                word_list.append(terms_list[index])
                j += 1
            else:
                break
        f.write(" ".join(word_list))
        f.write(" \n")
    f.close()

