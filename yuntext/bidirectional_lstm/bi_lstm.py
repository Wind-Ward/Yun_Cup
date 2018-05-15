# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model
sys.path.append("../")

from data_process import cfg, get_embedding_layer
from base_model import TextModel


class BiLSTM(TextModel):
    def get_model(self, trainable=None):
        trainable = self.trainable if trainable is None else trainable
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=trainable,
                                  use_new_vector=self.use_new_vector)(inputs)
        emb = SpatialDropout1D(0.7)(emb)
        x = Bidirectional(CuDNNLSTM(cfg.LSTM_hidden_size, return_sequences=True))(emb)
        x = Bidirectional(CuDNNLSTM(cfg.LSTM_hidden_size, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        # x = Dropout(0.5)(x)
        x = Dense(5, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))


class BiLSTM1(TextModel):
    def get_model(self, trainable=None):
        trainable = self.trainable if trainable is None else trainable
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=trainable,
                                  use_new_vector=self.use_new_vector)(inputs)
        emb = SpatialDropout1D(0.7)(emb)
        x = Bidirectional(CuDNNLSTM(cfg.LSTM_hidden_size, return_sequences=True))(emb)
        x = Bidirectional(CuDNNLSTM(cfg.LSTM_hidden_size, return_sequences=True))(x)
        max_pool = GlobalMaxPool1D()(x)
        avg_pool = GlobalAvgPool1D()(x)
        x = concatenate([max_pool, avg_pool], axis=1)
        # x = Dropout(0.5)(x)
        x = Dense(5, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))
