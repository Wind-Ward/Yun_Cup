import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pickle
import numpy as np
from sklearn.svm import LinearSVR


def get_data():
    train = pd.read_csv("../../data/train_22w_translate_v11.csv")
    test = pd.read_csv("../../data/test_5w_translate_v11.csv")
    print(train.shape)


    data = pd.concat([train, test])
    print('train %s test %s' % (train.shape, test.shape))
    print('train columns', train.columns)
    return data, train.shape[0], train['Score'], test['Id']

def xx_mse_s(y_true, y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res': list(y_pre)})

    #y_pre['res'] = y_pre['res'].astype(int)  # astype数据类型转换。  根本不需要啊这里！！
    return 1 / (1 + mean_squared_error(y_true, y_pre['res'].values) ** 0.5)


def pre_process(): #简单的特征工程
    data, nrw_train, y, test_id = get_data()

    tf = TfidfVectorizer(ngram_range=(1,5), analyzer="char")  # max_features=50000
    discuss_tf= tf.fit_transform(data['discuss_segment_jieba'])



    tf_1 = TfidfVectorizer(ngram_range=(2,5), analyzer="word")  # max_features=50000   #特征构造进行不断的尝试
    discuss_tf_1= tf_1.fit_transform(data['discuss_segment_jieba'])
    print(discuss_tf_1.shape[1])
    print( discuss_tf_1.shape[0])


    #特意增加维度，出现了显著的效果   三个hstack的维度(100000, 16866186)总维度
    data = hstack((discuss_tf,discuss_tf_1,discuss_tf_1)).tocsr()

    return data[:nrw_train], data[nrw_train:], y, test_id


#预测分值取整
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


if __name__ == '__main__':
    X, test, train_Y, test_id = pre_process()
    print("X shape:", X.shape)

    ch2=SelectKBest(chi2,7000000)  #升维度
    train_X=ch2.fit_transform(X,train_Y)
    test_X=ch2.transform(test)
    print("X new shape:",X.shape)

    kfold = 10
    folds = pickle.load(open('../../data/fold_10_train_220000_test_50000_by_ding_server.pkl','rb'))

    ridge = Ridge(solver='auto', fit_intercept=True, alpha=4, max_iter=1000, normalize=False, tol=0.01,
                    random_state=1)


    base_models = (ridge,)

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
            if model in ['LGBMClassifier', 'XGBClassifier', ]:
                clf.set_params(random_state=j)
                clf.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], early_stopping_rounds=100, eval_metric='rmse',
                        verbose=False)
            elif model == 'CatBoostClassifier':
                clf.fit(X_train, y_train, eval_set=[X_eval, y_eval], use_best_model=True, verbose=False)
            else:
                clf.fit(X_train, y_train)

            y_pred = clf.predict(X_eval)

            S_train[test_index, i] = y_pred  # 可以这个 i一直都是1 因为basemodel 只有一个， S_train必然就是一列
            S_test_i[:, j] = clf.predict(T)

            cv_pred.append(S_test_i[:, j])

            # 记录cv分数
            temp = xx_mse_s(y_eval, y_pred)
            mse.append(temp)
            temp3 = pd.DataFrame({'res': list(y_pred)})
            print("fold:", j)
            temp3["res"] = round_score(temp3["res"])
            temp2 = xx_mse_s(y_eval, list(temp3["res"]))
            mse_round.append(temp2)

        # 显示取阈值和没有取阈值的cv分数
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
            res.to_csv('./output/ridge_stacking.csv', index=False, header=False)
            res["pre"] = round_score(res["pre"])

            res.to_csv('./output/ridge_round_score_stacking.csv', index=False,
                       header=False)


        write()
        ## 保存元特征
    pickle.dump(S_train, open('./output/S_train_ridge.pkl', 'wb'))
    pickle.dump(S_test, open('./output/S_test_ridge.pkl', 'wb'))






