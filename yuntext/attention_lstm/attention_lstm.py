# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from data_process import get_embedding_layer
from base_model import TextModel


class AttentionLSTM(TextModel):
    def get_model(self, trainable=None):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=trainable,
                                  use_new_vector=self.use_new_vector)(inputs)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(emb)
        x = self.attention_3d_block(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))

    def attention_3d_block(self, inputs):
        """
        attention mechanisms for lstm
        :param inputs: shape (batch_size, seq_length, input_dim)
        :return:
        """
        a = Permute((2, 1))(inputs)
        a = Dense(self.max_len, activation='softmax')(a)
        a_probs = Permute((2, 1))(a)    # attention_vec
        att_mul = multiply([inputs, a_probs])
        return att_mul
