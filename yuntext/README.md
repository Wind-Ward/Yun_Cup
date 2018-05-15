## 配置环境
### 主要依赖:
    - keras, tensorflow, gensim, jieba, numpy, pandas, matplotlib, sklearn

### 在项目目录下面创建`input`文件夹. 该文件夹的情况
    .
    ├── YNU.EDU2018-ScenicWord
    │   ├── predict_first.csv
    │   └── train_first.csv
    ├── YNU.EDU2018-ScenicWord-Semi
    │   ├── predict_second.csv
    │   └── train_second.csv
    ├── YNU.EDU2018-ScenicWord_submite_sample.csv
    ├── pkl_dir
    │   └── fold_10_train_220000_test_50000_by_ding_server.pkl
    ├── processed
    └── word2vec
- `pkl_dir`里面存放的文件是一个十折交叉验证的list，创建该文件原因是便于stacking。(将base_model.py的81-82解注释, 83-85注释掉就是一个随机十折交叉)
- `YNU.EDU2018-ScenicWord`是初赛训练和测试数据
- `YNU.EDU2018-ScenicWord-Semi`是复赛训练和测试数据
- `YNU.EDU2018-ScenicWord_submite_sample.csv` 复赛提交文件样本
- processed, word2vec 分别用来存放处理后的数据和训练完的词向量的文件夹

### 每个模型的目录下面创建result目录

## 处理数据(按照顺序执行)
### 处理原始数据
进入data_process目录下执行`python data_process.py`

### 训练word2vec
在data_process目录下执行`python w2v.py`


## 训练模型
#### textcnn
```
python train.py --classifier=textcnn.TextCNNMultiKernel1D
```
该模型阈值取值是将2.0一下的调整为1.0, 将4.7以上的调整为5.0

#### textrcnn
```
python train.py --classifier=textrcnn.TextRCNN
```
该模型阈值取值是1.5一下的调整为1.0, 将4.75以上的调整为5.0

#### attention_lstm
```
python train.py --classifier=attention_lstm.AttentionLSTM1_c1
```
该模型阈值同textrcnn

#### capsule_lstm
```
python train.py --classifier=bidirectional_lstm.CapLSMT
python train.py --classifier=bidirectional_lstm.CapLSMT_c7
```
模型阈值同textrcnn

#### textrnn
```
python train.py --classifier=bidirectional_lstm.BiLSTM
python train.py --classifier=bidirectional_lstm.BiLSTM1
```
阈值同textrcnn
