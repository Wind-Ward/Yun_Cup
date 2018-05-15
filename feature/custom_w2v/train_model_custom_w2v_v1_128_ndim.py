import pandas as pd
from gensim.models import word2vec
import smart_open
import gensim
import numpy as np

train_data =['./output/train_segment_220000_v11_modify_processed_stopword_translate_modify.txt',"./output/test_segment_50000_v11_modify_processed_stopword_translate_modify.txt"]
def read_corpus(train_data):
    temp=[]
    for source_file in train_data:   # 并别读入

        with smart_open.smart_open(source_file,encoding='utf-8') as f:
            for i,line in enumerate(f) :
                temp.append(gensim.utils.to_unicode(line).split())

    return temp


# 1. load tagged corpus
train_corpus =read_corpus(train_data)
model=word2vec.Word2Vec(train_corpus, min_count=1, workers=12,size=128)
model.save('./word2vec_v1.model')


train_num=220000
test_num=50000
total=train_num+test_num
train_arrays=np.zeros((train_num,128))
test_arrays=np.zeros((test_num,128))
for index,item in enumerate(train_corpus):
    if index < train_num:

        for word in item:
            train_arrays[index]+= model[word]
        train_arrays[index]/=len(item)
    elif index < total:
        for word in item:
            test_arrays[index-train_num]+= model[word]
        test_arrays[index-train_num]/=len(item)

t=pd.DataFrame(train_arrays)
t.to_csv("./output/train_custom_word2vec_v1.csv",index=False,header=None)

t=pd.DataFrame(test_arrays)
t.to_csv("./output/test_custom_word2vec_v1.csv",index=False,header=None)

