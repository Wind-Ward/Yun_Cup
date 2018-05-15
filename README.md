# "Yun Cup" Scenic Reputation Evaluation Score Forecast

### Introduction
This package includes 3th solution for the ["Yun Cup" Scenic Reputation Evaluation Score Forecast](http://www.datafountain.cn/#/competitions/283/intro).

![text](https://github.com/Wind-Ward/Yun_Cup/raw/master/vendor/1.png)


### Directory
- `model`: machine learning model & deel learning model meta feature for stacking purpose.
- `preprocess`: preprocesss for machine learning model.
- `stacking`: stacking model.
- `yuntext`: deep learning model(including detailed instructions to setup).


### Ensemble
* Stacking get better performence in LB.
![text](https://github.com/Wind-Ward/Yun_Cup/raw/master/vendor/2.png)


### Score
|model|score|
:---:|:----:
FastText|0.54018 (pretrained embedding)
Ridge|0.54449
Select-K-Best|~0.543
Word2vec|0.549
CNN|0.556
RCNN|0.555
Capsule|0.549
HAN(LSTM-Attention)|0.550
RNN|0.547

### Failed
* Data Augment
* [***TF-IDF-CD***](http://manu44.magtech.com.cn/Jwk_infotech_wk3/article/2015/1003-3513/1003-3513-31-3-39.html)
* Crawl comments from scenic reputation website to pretrain word embeddings.
* Pseudo-Labelling


### Reference
* [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/kernels)
* ***Large Scale Multi-label Text Classification With Deep Learning***
* ***Convolutional Neural Networks for Sentence Classification***
* ***Convolutional Neural Networks for Sentence Classification***
* ***Recurrent Convolutional Neural Networks for Text Classification***
* ***Neural Machine Translation of Rare Words with Subword Units***
* ***A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification***

### Acknowledgments
* [WindWard](https://github.com/Wind-Ward)
* [fpc](https://github.com/stanpcf)

