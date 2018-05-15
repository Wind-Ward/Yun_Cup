#!/usr/bin/env python
# coding: utf8

import os
import re
from tqdm import tqdm
import pandas as pd

# 分词工具
import jieba

from utils import get_stop_words

stop_words = get_stop_words()
fill_value = "CSFxe"
# user_dict = './yan_word.txt'

def clean_str(stri):
    stri = re.sub(r'[a-zA-Z0-9]+', '', stri)
    if stri == '':
        return fill_value
    return stri.strip()


def _filter_stop_words(word_list):
    _filter_words = [w for w in word_list if w not in stop_words and len(w) > 0]
    x = " ".join(_filter_words)
    return x


data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
processed_dir = "processed"
train_file = 'train_second.csv'
test_file = 'predict_second.csv'

train_a = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', 'train_first.csv'))
train_b = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord-Semi', train_file))
train = pd.concat([train_a, train_b], ignore_index=True)
print("--->train shape:", train.shape)
test_a = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', 'predict_first.csv'))
test = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord-Semi', test_file))

jieba_train, jieba_test, jieba_test_a = [], [], []

print("process train data")
for text in tqdm(train["Discuss"].values):
    jieba_train.append(_filter_stop_words(jieba.cut(clean_str(text))))

print("process test data")
for text in tqdm(test["Discuss"].values):
    jieba_test.append(_filter_stop_words(jieba.cut(clean_str(text))))

print("process test data")
for text in tqdm(test_a["Discuss"].values):
    jieba_test_a.append(_filter_stop_words(jieba.cut(clean_str(text))))

train['jieba'] = jieba_train
test['jieba'] = jieba_test
test_a['jieba'] = jieba_test_a

for col in ['Discuss', 'jieba']:
    train[col].fillna(value=fill_value, inplace=True)
    test[col].fillna(value=fill_value, inplace=True)


_processed_dir = os.path.join(data_dir, processed_dir)
if not os.path.exists(_processed_dir):
    os.mkdir(_processed_dir)

train.to_csv(os.path.join(data_dir, processed_dir, train_file), index=False)
test.to_csv(os.path.join(data_dir, processed_dir, test_file), index=False)
test_a.to_csv(os.path.join(data_dir, processed_dir, 'predict_first.csv'), index=False)
