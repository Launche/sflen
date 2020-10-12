import os
import sys

import pandas as pd
from deepctr.feature_column import SparseFeat, get_feature_names, DenseFeat
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf

from deepctr.models import *

from constant import *
from data_source import get_data
from plot_curves import history_curves

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 10
    if sys.argv.__len__() == 3:
        data_type = sys.argv[1]
        epochs = int(sys.argv[2])

    data, train, test = get_data(data_type)

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    # use_bn=False mean that it not use bn after ffm out
    model = ONN(linear_feature_columns, dnn_feature_columns, task='binary', use_bn=False, dnn_dropout=0.5)
    model.compile("adam", "binary_crossentropy",
                  metrics=METRICS)

    log_dir = './log/ffm_' + data_type + '_' + str(epochs)
    if not os.path.exists(log_dir):  # 如果路径不存在
        os.makedirs(log_dir)

    logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[logs])
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    history_curves(history)
