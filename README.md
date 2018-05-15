# "Yun Cup" Scenic Reputation Evaluation Score Forecast

### Introduction
This package includes 3th solution for the ["Yun Cup" Scenic Reputation Evaluation Score Forecast] (http://www.datafountain.cn/#/competitions/283/intro).

![text](https://github.com/Wind-Ward/Yun_Cup/raw/master/vendor/1.png)


### Directory
- `model`: machine learning model & deel learning model meta feature for stacking purpose.
- `preprocess`: preprocesss for machine learning model.
- `stacking`: stacking model.
- `yuntext`: deep learning model(including detailed instructions to setup).


### Machine Learning
![text](https://github.com/Wind-Ward/Yun_Cup/raw/master/vendor/3.png)


### Deep Learnging
![text](https://github.com/Wind-Ward/Yun_Cup/raw/master/vendor/4.png)


### Stacking 
* Stacking ensemble is much better than blending in our models.
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

### Reference


