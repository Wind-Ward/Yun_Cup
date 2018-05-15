# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from data_process import get_embedding_layer, cfg
from base_model import TextModel, Attention


class AttentionLSTM1(TextModel):
    def get_model(self, trainable=None):
        trainable = self.trainable if trainable is None else trainable
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=trainable,
                                  use_new_vector=self.use_new_vector)(inputs)
        emb = SpatialDropout1D(0.5)(emb)
        x = Bidirectional(CuDNNGRU(cfg.LSTM_hidden_size, return_sequences=True))(emb)
        x = Bidirectional(CuDNNGRU(cfg.LSTM_hidden_size, return_sequences=True))(x)
        x = Attention(self.max_len)(x)
        x = Dropout(0.5)(x)
        x = Dense(5, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))
