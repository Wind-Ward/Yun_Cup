# -*- coding: utf-8 -*-

import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from easydict import EasyDict
try:
    from .config import cfg
except ModuleNotFoundError as e:
    from config import cfg

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def get_data(max_len=80):
    """
    :param max_len: 给每个句子设定的长度，多截少填
    :return: x.shape=(sample, max_len), every sentence is sequenced.

    """
    fill_value = "CSFxe"
    ctp = "jieba"
    train = pd.read_csv(os.path.join(data_dir, "processed", "train_second.csv"))    # 里面包含了初赛的训练数据
    test = pd.read_csv(os.path.join(data_dir, "processed", "predict_second.csv"))

    train["Discuss"].fillna(value=fill_value, inplace=True)
    train[ctp].fillna(value=fill_value.lower(), inplace=True)
    test["Discuss"].fillna(value=fill_value, inplace=True)
    test[ctp].fillna(value=fill_value.lower(), inplace=True)

    tokenizer = Tokenizer(num_words=cfg.MAX_FEATURE)
    tokenizer.fit_on_texts(train[ctp].values)

    x_train, y_train = train[ctp].values, train["Score"].values
    x_test = test[ctp].values

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    data = EasyDict({'x_train': x_train, 'y_train': y_train,
                     # 'x_valid': x_valid, 'y_valid': y_valid, 'valid_id': dvalid['Id'].values,
                     'x_test': x_test, 'test_id': test["Id"].values,
                     'tokenizer': tokenizer})
    return data


if __name__ == '__main__':
    data = get_data()
