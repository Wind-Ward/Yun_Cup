#!/usr/bin/env python
# coding: utf8

from pathlib import Path
import multiprocessing
import logging
import pandas as pd
from gensim.models import Word2Vec
from tensorflow import flags

from utils import get_stop_words

flags.DEFINE_integer('hidden_dim', 200, 'hidden dim of word2vec')
flags.DEFINE_integer('iter', 50, 'iter number')
flags.DEFINE_integer('min_count', 3, 'filter number when less than min_count')
flags.DEFINE_integer('window', 5, 'window of context')

FLAGS = flags.FLAGS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train():
    pro_dir = Path(__file__).absolute().parent.parent
    train_file = pro_dir / 'input' / 'processed' / 'train_second.csv'
    test_file = pro_dir / 'input' / 'processed' / 'predict_second.csv'
    test_a_file = pro_dir / 'input' / 'processed' / 'predict_first.csv'
    columns = ['jieba']

    stop_words = get_stop_words()
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    test_a = pd.read_csv(test_a_file)

    corpus = []
    for df in [train, test, test_a]:
        for col in columns:
            ser = df[col]
            ser = ser[ser.isnull() == False]  # 因为样本中有NAN
            v = ser.apply(lambda x: [w for w in x.split(" ") if w not in stop_words]).values.tolist()
            corpus.extend(v)

    model = Word2Vec(corpus, size=FLAGS.hidden_dim, window=FLAGS.window, min_count=len(columns) * FLAGS.min_count,
                     sg=1, iter=FLAGS.iter, workers=multiprocessing.cpu_count())

    vector_file = pro_dir / 'input' / 'word2vec' / "my_w2v_{dim}_{iter}_{wd}.txt".format(
        dim=FLAGS.hidden_dim, iter=FLAGS.iter, wd=FLAGS.window)
    vector_file.parent.mkdir(mode=0o755, exist_ok=True)

    model.wv.save_word2vec_format(str(vector_file), binary=False)


if __name__ == '__main__':
    train()
