import pandas as pd
import numpy as np
import jieba
import re
from sklearn.model_selection import StratifiedKFold,KFold
import random
import fasttext
import pickle
from sklearn.metrics import mean_squared_error

#直接用预处理好的文本
df = pd.read_csv('../../data/train_22w_translate_v11.csv')
test_df = pd.read_csv("../../data/test_5w_translate_v11.csv")

def fasttext_data(data,label):
    fasttext_data = []
    for i in range(len(label)):
        sent = data[i]+"\t__label__"+str(int(label[i]))
        fasttext_data.append(sent)
    with open('train_wl.txt','w') as f:
        for data in fasttext_data:
            f.write(data)
            f.write('\n')
    return 'train_wl.txt'





def round_score(data):
    count = [0, 0, 0, 0, 0]

    def _round_score(score):
        if score > 4.7:
            count[4] += 1
            return 5.0

        else:
            return score
    data = data.apply(lambda x: _round_score(x))
    print(count)
    return data


def xx_mse_s(y_true, y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res': list(y_pre)})
    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / (1 + mean_squared_error(y_true, y_pre['res'].values) ** 0.5)


def compute_score(result_pred):
    prob = []
    score = []
    for sco, pr in result_pred:
        prob.append(pr)
        score.append(int(sco))
    prob = np.array(prob)
    score = np.array(score)
    prob = prob / prob.sum()
    return sum(prob * score)


def fast_cv(df):
    df = df.reset_index(drop=True)
    X = df['discuss_segment_jieba'].values
    y = df['Score'].values
    T = test_df['discuss_segment_jieba'].values
    fast_pred = []

    kfold = 10

    S_test_i = np.zeros((T.shape[0], kfold))
    S_train = np.zeros((df.shape[0], 1))
    S_test = np.zeros((test_df.shape[0], 1))

    folds = pickle.load(open('../../data/fold_10_train_220000_test_50000_by_ding_server.pkl','rb'))

    mse = []
    mse_round = []
    cv_pred = []
    for j, (train_index, test_index) in enumerate(folds):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_file = fasttext_data(X_train, y_train)

        classifier = fasttext.supervised(train_file, 'fasttext_v3.model', lr=0.02, dim=300, epoch=60, ws=7,
                                         loss='softmax', min_count=3, word_ngrams=5, bucket=2000000,
                                         label_prefix="__label__",pretrained_vectors="./output/wiki.zh.vec")
#对验证集进行预测：

        result = classifier.predict_proba(df.loc[test_index, 'discuss_segment_jieba'].tolist(), k=5)
        pred = [compute_score(result_i) for result_i in result]
        S_train[test_index, 0] = pred

        temp = xx_mse_s(y_test, pred)
        mse.append(temp)
        temp3 = pd.DataFrame({'res': list(pred)})
        temp3["res"] = round_score(temp3["res"])
        temp2 = xx_mse_s(y_test, list(temp3["res"]))
        mse_round.append(temp2)
        print("fold_{}_cv_original_score_{}".format(j + 1, temp))
        print("fold_{}_cv_round_score_{}".format(j + 1, temp2))

#对全测试集进行预测：

        test_result = classifier.predict_proba(test_df['discuss_segment_jieba'].tolist(), k=5)
        fast_predi = [compute_score(result_i) for result_i in test_result]

        S_test_i[:, j]= fast_predi

        fast_pred.append(fast_predi)

    print('cv_original_result:{}'.format(np.mean(mse)))
    print('cv_round_result:{}'.format(np.mean(mse_round)))
    print("-" * 20)

    fast_pred = np.array(fast_pred)
    fast_pred = np.mean(fast_pred, axis=0)

    S_test[:, 0] = S_test_i.mean(axis=1)

    return fast_pred ,S_train, S_test


data = np.zeros((len(test_df),2))
sub_df = pd.DataFrame(data)
sub_df.columns = ['Id','mean']
sub_df['Id'] = test_df['Id'].values

test_pred,S_train, S_test= fast_cv(df)
sub_df['mean'] = test_pred.copy()

pred = test_pred.copy()    #对预测结果进行取阈值 统一一下
pred = np.where(pred>4.7,5,pred)
pred = np.where(pred<1, 1, pred)
sub_df['mean'] = pred

from datetime import datetime
subfix = datetime.now().strftime('%y%m%d%H%M')
submiss=sub_df[['Id','mean']]
submiss.to_csv('./output/fasttext_'+ subfix + '.csv',header=None,index=False)
pickle.dump(S_train, open('./output/S_train_fasttext_v3_pretrained.pkl', 'wb'))
pickle.dump(S_test, open('./output/S_test_fasttext_v3_pretrained.pkl', 'wb'))
