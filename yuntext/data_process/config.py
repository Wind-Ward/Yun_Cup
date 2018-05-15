# coding: utf8


class cfg:
    """
    该类保存一些全局的网络配置
    """
    MAX_FEATURE = 200000

    # 样本的权重, 索引代表类别, 索引位置的值代表该类别权重 [0, 100, 60, 6, 2, 1]
    # SAMPLE_WEIGHT = [8, 8, 4, 1.4, 1]    # [10, 6, 3, 1.2, 1] [15, 6.5, 4, 1.2, 1]

    MY_EMBED_WIND = 5   # 该变量保存的是自己训练的词向量的context窗口长度
    MY_EMBED_SIZE_CHOICE = {100, 200, 300}      # 可选的词向量维度

    MODEL_FIT_validation_split = 0.1            # model.fit, keras

    TRAIN_TEST_SPLIT_random_state = 123         # train_test_split,  sklearn

    TEXT_CNN_filters = [1, 2, 3, 4, 5]    # text cnn 多卷积通道的卷积核大小

    LSTM_hidden_size = 128

    Keras_padding = "post"          # pre,     keras, pad_sequence的方式.

    top_k = 2

    enhance_to = [3092, 7223, 12513, 32580, 66030]
    # enhance_to = [3092, 7223, 12513, 32580, 66030]
