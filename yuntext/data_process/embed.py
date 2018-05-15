# coding: utf8

import os
from pathlib import Path
import numpy as np
from keras.layers import Embedding

from .config import cfg


def get_embedding_layer(tokenizer, max_len=100, embedding_dim=100, use_pretrained=True, trainable=True,
                        use_new_vector=False):
    """
    :param tokenizer: keras.prerpocessing.text.Tokenizer
    :param max_len:
    :param embedding_dim:
    :param use_pretrained: if use pretrained vector to init embedding layer. default true
    :param trainable:
    :param use_new_vector:
    :return: keras embedding layer
    """
    use_new_vector = True
    assert embedding_dim in cfg.MY_EMBED_SIZE_CHOICE
    word_index = tokenizer.word_index
    # num_words = min(cfg.MAX_FEATURE, len(word_index))
    num_words = min(cfg.MAX_FEATURE, len(word_index)+1)

    if not use_pretrained:
        return Embedding(num_words, embedding_dim, input_length=max_len)

    initial_weights = _get_glove_weights(num_words, word_index, embedding_dim, use_new_vector)
    return Embedding(num_words, embedding_dim, weights=[initial_weights], input_length=max_len, trainable=trainable)


def _get_glove_weights(num_words, word_index, dim, use_new_vector):
    """
    :param word_index: token.word_index
    :param dim: 维度
    :return: embedding 权重
    """
    embeddings_index = {}
    _w2v_dir = 'word2vec' if use_new_vector else 'word2vec_bak'
    W2V_DIR = Path(__file__).absolute().parent.parent / 'input' / _w2v_dir
    # filename = "wiki.zh.vec"
    filename = "my_w2v_{dim}_50_{_wind}.txt".format(dim=dim, _wind=cfg.MY_EMBED_WIND)
    filename = os.path.join(W2V_DIR, filename)
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((num_words, dim))
    for word, i in word_index.items():
        if i >= cfg.MAX_FEATURE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
