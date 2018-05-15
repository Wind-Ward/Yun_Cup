import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def xx_mse_s(y_true, y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res': list(y_pre)})

    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / (1 + mean_squared_error(y_true, y_pre['res'].values) ** 0.5)


#修改阈值的函数
def round_score(data):
    count = [0, 0, 0, 0, 0]
    def _round_score(score):
        if score > 4.7:
            count[4] += 1
            return 5.0

        if score <1:
            count[0]+=1
            return 1
        return score
    data=data.apply(lambda x:_round_score(x))
    print(count)
    return data



#kaggle_select_K_feature
select_K_feature_train_pkl=pickle.load(open("../model/Select_K_Best/output/stacking_cache/S_train_LightGBM_with_Select_K_Best_on_TFIDF_stacking_v2.pkl","rb"))
select_K_feature_test_pkl=pickle.load(open("../model/Select_K_Best/output/stacking_cache/S_test_LightGBM_with_Select_K_Best_on_TFIDF_stacking_V2.pkl","rb"))

#select K xgb
select_K_xgb_cd_train=pickle.load(open("../model/Select_K_Best/output/stacking_cache/S_train_XGB_with_Select_K_Best_on_TFIDF_stacking_v3.pkl","rb"))
select_K_xgb_cd_test=pickle.load(open("../model/Select_K_Best/output/stacking_cache/S_test_XGB_with_Select_K_Best_on_TFIDF_stacking_v3.pkl","rb"))


#ridge
ridge_train_pkl=pickle.load(open("../model/ridge/output/stacking_cache/S_train_ridge.pkl","rb"))
ridge_test_pkl=pickle.load(open("../model/ridge/output/stacking_cache/S_test_ridge.pkl","rb"))


#w2v 200dim
w2v_200_train_lgb=pickle.load(open("../model/w2v/output/stacking_cache/S_train_custom_w2v_220000_lgb_v5_200_ndim_stacking.pkl","rb"))
w2v_200_test_lgb=pickle.load(open("../model/w2v/output/stacking_cache/S_test_custom_w2v_50000_lgb_v5_200_ndim_stacking.pkl","rb"))



#w2v 200dim

w2v_200_train_xgb=pickle.load(open("../model/w2v/output/stacking_cache/S_train_custom_w2v_220000_xgb_v5_200_ndim_stacking.pkl","rb"))
w2v_200_test_xgb=pickle.load(open("../model/w2v/output/stacking_cache/S_test_custom_w2v_50000_xgb_v5_200_ndim_stacking.pkl","rb"))


#w2v 128ndim
w2v_128_train_lgb=pickle.load(open("../model/w2v/output/stacking_cache/S_train_custom_w2v_220000_lgb_v5_stacking.pkl","rb"))
w2v_128_test_lgb=pickle.load(open("../model/w2v/output/stacking_cache/S_test_custom_w2v_50000_lgb_v5_stacking.pkl","rb"))


#fasttext
fast_train_pkl=pickle.load(open("../model/fasttext/output/stacking_cache/S_train_fasttext_v3_pretrained.pkl","rb"))
fast_test_pkl=pickle.load(open("../model/fasttext/output/stacking_cache/S_test_fasttext_v3_pretrained.pkl","rb"))


#attention
attention_train=pickle.load(open("../model/attention/AttentionLSTM1_train_stacking.pkl","rb"))
attention_test=pickle.load(open("../model/attention/AttentionLSTM1_test_stacking.pkl","rb"))

#capsule
capsule_train=pickle.load(open("../model/capsule/CapLSMT_train_stacking.pkl","rb"))
capsule_test=pickle.load(open("../model/capsule/CapLSMT_test_stacking.pkl","rb"))


#cnn
cnn_train=pickle.load(open("../model/cnn/TextCNNMultiKernel1D_train_stacking.pkl","rb"))
cnn_test=pickle.load(open("../model/cnn/TextCNNMultiKernel1D_test_stacking.pkl","rb"))


#rcnn
rcnn_train=pickle.load(open("../model/rcnn/TextRCNN_train_stacking.pkl","rb"))
rcnn_test=pickle.load(open("../model/rcnn/TextRCNN_test_stacking.pkl","rb"))

#rnn
rnn_train=pickle.load(open("../model/rnn/BiLSTM_train_stacking.pkl","rb"))
rnn_test=pickle.load(open("../model/rnn/BiLSTM_test_stacking.pkl","rb"))




#将元特征合并起来
S_train=np.column_stack((select_K_feature_train_pkl,select_K_xgb_cd_train,ridge_train_pkl,w2v_200_train_lgb,w2v_200_train_xgb,w2v_128_train_lgb,fast_train_pkl,attention_train,capsule_train,cnn_train,rcnn_train,rnn_train))
S_test=np.column_stack((select_K_feature_test_pkl,select_K_xgb_cd_test,ridge_test_pkl,w2v_200_test_lgb,w2v_200_test_xgb,w2v_128_test_lgb,fast_test_pkl,attention_test,capsule_test,cnn_test,rcnn_test,rnn_test))





train_a=pd.read_csv("../data/train_first.csv")
train_b=pd.read_csv("../data/train_second.csv")
train=pd.concat([train_a,train_b],ignore_index=True)
train_Y=train["Score"]
submit=pd.read_csv("../data/predict_second.csv")
submit=submit[["Id"]]
submit["Score"]=0

folds = pickle.load(open('../data/fold_10_train_220000_test_50000_by_ding_server.pkl','rb'))

params = {'learning_rate': 0.001,
          'objective': 'regression',
          'metric': 'rmse',
          'seed':1,
          'num_threads':8,
          }

kfold=10
mse = []  # 每折得分保存
mse_round = []
for i, (train_index, test_index) in enumerate(folds):
    print('kfold: {}  of  {} : '.format(i + 1, kfold))
    params['seed'] = i
    X_train, X_eval = S_train[train_index], S_train[test_index]
    y_train, y_eval = train_Y[train_index], train_Y[test_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_eval, label=y_eval)
    lgb_model = lgb.train(params, dtrain, 20000, dval, verbose_eval=200,early_stopping_rounds=100)

    y_pred=lgb_model.predict(X_eval)
    # 记录cv分数
    temp = xx_mse_s(y_eval, y_pred)
    mse.append(temp)
    temp3 = pd.DataFrame({'res': list(y_pred)})
    temp3["res"] = round_score(temp3["res"])
    temp2 = xx_mse_s(y_eval, list(temp3["res"]))
    mse_round.append(temp2)
    submit['Score'] += lgb_model.predict(S_test) / kfold
    print("fold_{}_original_score:{}".format(i,temp))
    print("fold_{}_round_score:{}".format(i,temp2))

print('cv_original_result:{}'.format(np.mean(mse)))
print('cv_round_result:{}'.format(np.mean(mse_round)))
print("-" * 20)
submit.to_csv('./output/final_result.csv',index=False, header=False)
submit["Score"]=round_score(submit["Score"])
submit.to_csv('./output/final_result_round_score.csv',index=False, header=False)
print("S_train shape:",S_train.shape)









