import pandas as pd
import re
import jieba
from snownlp import SnowNLP


def jian(text):  #将繁体字转化为简体字
    w=SnowNLP(text)
    return w.han


train_a=pd.read_csv("../data/train_first.csv")
train_b=pd.read_csv("../data/train_second.csv")
train=pd.concat([train_a,train_b],ignore_index=True)
test=pd.read_csv("../data/predict_second.csv")
test_a=pd.read_csv("../data/predict_first.csv")

def add_stop_words(file_name="../data/stop_words_ch1_v3.txt"):
    stop_words = [" "]
    with open(file_name, "r") as f:
        for item in f.readlines():
            stop_words.append(item.strip())
    return set(stop_words)
stop_word = add_stop_words()


def clean_str(stri):
    stri = re.sub('[a-zA-Z0-9\s]+',' ',stri)
    cut_str = jieba.cut(stri.strip())
    list_str = [word for word in cut_str if word not in stop_word]
    stri = ' '.join(list_str)
    return stri


def fillnull(x):
    if x == '':
        return '<UNK>'
    else:
        return x

train_translate=pd.read_csv("./data/train_22w_non_Chinese_translate_654.csv")
test_translate=pd.read_csv("./data/test_5w_non_Chinese_translate_137.csv")


test_a_translate=pd.read_csv("./data/test_3w_non_Chinese_translate_90.csv")
train=pd.merge(train,train_translate[["Id","discuss_translate"]],on="Id",how="left")
test=pd.merge(test,test_translate[["Id","discuss_translate"]],on="Id",how="left")

test_a=pd.merge(test_a,test_a_translate[["Id","discuss_translate"]],on="Id",how="left")

train_translate_list=[]
train.fillna("",inplace=True)
for index,item in train.iterrows():
    if item["discuss_translate"]=="":
        train_translate_list.append(item["Discuss"])
    else:
        train_translate_list.append(item["discuss_translate"])

train["discuss_translate"]=train_translate_list
train["discuss_translate"]=train["discuss_translate"].apply(lambda x:jian(x))
train["discuss_segment_jieba"]=train["discuss_translate"].apply(lambda x:clean_str(x))
train["discuss_segment_jieba"]=train["discuss_segment_jieba"].apply(lambda x:fillnull(x))
train.to_csv("train_22w_translate_v11.csv",index=False)
#把生成的csv文件复制到根目录下的data目录下即可




test_translate_list=[]
test.fillna("",inplace=True)
for index,item in test.iterrows():
    if item["discuss_translate"]=="":
        test_translate_list.append(item["Discuss"])
    else:
        test_translate_list.append(item["discuss_translate"])

test["discuss_translate"]=test_translate_list
test["discuss_translate"]=test["discuss_translate"].apply(lambda x:jian(x))
test["discuss_segment_jieba"]=test["discuss_translate"].apply(lambda x:clean_str(x))
test["discuss_segment_jieba"]=test["discuss_segment_jieba"].apply(lambda x:fillnull(x))
test.to_csv("test_5w_translate_v11.csv",index=False)
#把生成的csv文件复制到根目录下的data目录下即可






test_translate_list=[]
test_a.fillna("",inplace=True)
for index,item in test_a.iterrows():
    if item["discuss_translate"]=="":
        test_translate_list.append(item["Discuss"])
    else:
        test_translate_list.append(item["discuss_translate"])


test_a["discuss_translate"]=test_translate_list
test_a["discuss_translate"]=test_a["discuss_translate"].apply(lambda x:jian(x))
test_a["discuss_segment_jieba"]=test_a["discuss_translate"].apply(lambda x:clean_str(x))
test_a["discuss_segment_jieba"]=test_a["discuss_segment_jieba"].apply(lambda x:fillnull(x))




test_a.to_csv("test_a_3w_translate_v11.csv",index=False)
#把生成的csv文件复制到根目录下的data目录下即可
