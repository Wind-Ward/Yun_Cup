import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import lightgbm as lgb
import jieba
from snownlp import SnowNLP
import re
import pickle
from lightgbm import LGBMRegressor
import numpy as np
from xgboost import XGBRegressor

word_vec_columns_name = []
for i in range(200):
    word_vec_columns_name.append("w2v_{}".format(i+1))


#读取w2v的csv文件
def merge_word2vec_feature(train,test):
    train_word2vec = pd.read_csv("../../feature/custom_w2v/output/train_22w_custom_word2vec_v2_200_ndim.csv", header=None)
    test_word2vec = pd.read_csv("../../feature/custom_w2v/output/test_5w_custom_word2vec_v2_200_ndim.csv", header=None)

    train_word2vec.columns = word_vec_columns_name
    test_word2vec.columns = word_vec_columns_name

    train = pd.concat([train, train_word2vec], axis=1)
    test = pd.concat([test, test_word2vec], axis=1)
    return train, test


def get_data():
    train_a= pd.read_csv("../../data/train_first.csv")
    train_b = pd.read_csv("../../data/train_second.csv")
    train=pd.concat([train_a,train_b],ignore_index=True)
    test = pd.read_csv("../../data/predict_second.csv")
    train, test = merge_word2vec_feature(train, test)
    print("train_shape",train.shape)
    print("test_shape", test.shape)
    data = pd.concat([train, test])
    return data, train.shape[0], train['Score'], test['Id']



def xx_mse_s(y_true, y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res': list(y_pre)})
    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / (1 + mean_squared_error(y_true, y_pre['res'].values) ** 0.5)




def get_feature(train):

    Features=[]
    Features.extend(word_vec_columns_name)
    df_new_data=train[Features]
    return df_new_data


def round_score(data):
    count = [0, 0, 0, 0, 0]

    def _round_score(score):
        if score > 4.7:
            count[4] += 1
            return 5.0
        if score<1.0:
            count[0]+=1
            return 1.0
        return score
    data = data.apply(lambda x: _round_score(x))
    print(count)
    return data


def pre_process():  # 简单的特征工程
    data, nrw_train, y, test_id = get_data()
    data = get_feature(data)


    return data[:nrw_train], data[nrw_train:], y, test_id


if __name__ == '__main__':
    X, test, y, test_id = pre_process()

    train_X,test_X,train_Y,test_id= X, test, y, test_id
    #
    print("X shape:",X.shape)
    print("test shape:",y.shape)
    train_X=train_X.values
    train_Y=train_Y.values
    test_X = test_X.values
    xgb_params = {
        'booster': 'gbtree',
        'n_estimators': 20000,
        'n_jobs': 36,
        "seed": 1,
        'max_depth':9,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'eta': 0.03,

    }

    kfold = 10
    xgb = XGBRegressor(**xgb_params)
    base_models = (xgb,)
    folds = pickle.load(open('../../data/fold_10_train_220000_test_50000_by_ding_server.pkl', 'rb'))

    S_train = np.zeros((train_X.shape[0], len(base_models)))
    S_test = np.zeros((test_X.shape[0], len(base_models)))

    for i, clf in enumerate(base_models):
        model = str(clf).split('(')[0]
        if len(model) > 40:
            model = str(clf).split('.')[2].split(' ')[0]
        print('Running {}'.format(model))
        X = train_X.copy()
        y = train_Y.copy()
        T = test_X.copy()
        S_test_i = np.zeros((T.shape[0], kfold))
        mse = []
        mse_round = []
        cv_pred = []
        for j, (train_index, test_index) in enumerate(folds):
            X_train, X_eval = X[train_index], X[test_index]
            y_train, y_eval = y[train_index], y[test_index]
            if model in ['LGBMRegressor', 'XGBRegressor',]:
                clf.set_params(random_state=j)
                clf.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], early_stopping_rounds=100, eval_metric='rmse',
                        verbose=200)
            elif model == 'CatBoostRegressor':
                clf.fit(X_train, y_train, eval_set=[X_eval, y_eval], use_best_model=True, verbose=False)
            else:
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_eval)
            S_train[test_index, i] = y_pred
            S_test_i[:, j] = clf.predict(T)

            cv_pred.append(S_test_i[:, j])

            temp = xx_mse_s(y_eval, y_pred)
            mse.append(temp)
            temp3 = pd.DataFrame({'res': list(y_pred)})
            temp3["res"] = round_score(temp3["res"])
            temp2 = xx_mse_s(y_eval, list(temp3["res"]))
            mse_round.append(temp2)
            print("fold_{}_cv_original_score_{}".format(j+1,temp))
            print("fold_{}_cv_round_score_{}".format(j+1,temp2))


        print('cv_original_result:{}'.format(np.mean(mse)))
        print('cv_round_result:{}'.format(np.mean(mse_round)))
        print("-" * 20)
        S_test[:, i] = S_test_i.mean(axis=1)

        def write():
            s = 0
            for i in cv_pred:
                s = s + i

            s = s / kfold
            res = pd.DataFrame()
            res['Id'] = list(test_id)
            res['pre'] = list(s)
            res.to_csv('./output/xgb_custom_w2v_v5_200_ndim_stacking.csv', index=False, header=False)
            res["pre"] = round_score(res["pre"])

            res.to_csv('./output/xgb_custom_w2v_v5_200_ndim_round_score_stacking.csv', index=False, header=False)


        write()
    ## 保存元特征
    ## S_train_v1的维度要和S_test_v1的一致
    pickle.dump(S_train, open('./output/S_train_custom_w2v_220000_xgb_v5_200_ndim_stacking.pkl', 'wb'))
    pickle.dump(S_test, open('./output/S_test_custom_w2v_50000_xgb_v5_200_ndim_stacking.pkl', 'wb'))
    print("X shape:", X.shape)

