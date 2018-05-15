#coding=utf-8


import jieba,os
import pandas as pd
import re




train=pd.read_csv("../../data/train_22w_translate_v11.csv")

test=pd.read_csv("../../data/test_5w_translate_v11.csv")

test_a=pd.read_csv("../../data/test_a_3w_translate_v11.csv")
print(train["Score"].value_counts())
print("train_len:",len(train))


train["discuss_segment_jieba"].to_csv("./output/train_segment_220000_v11_modify_processed_stopword_translate_modify.txt",header=None,index=False)
test["discuss_segment_jieba"].to_csv("./output/test_segment_50000_v11_modify_processed_stopword_translate_modify.txt",header=None,index=False)
test_a["discuss_segment_jieba"].to_csv("./output/test_segment_a_30000_v11_modify_processed_stopword_translate_modify.txt",header=None,index=False)



