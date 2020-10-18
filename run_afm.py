import os
import sys

from deepctr.models import *
from deepctr.feature_column import SparseFeat, get_feature_names
from constant import *
from utils import get_data, output

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 100
    optimizer = "adam"
    dropout = 0.5
    data_type = 'enc'
    if sys.argv.__len__() == 3:
        data_type = sys.argv[1]
        epochs = int(sys.argv[2])

    data, train, test = get_data(data_type)

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    # use_attention=False mean that it is the same as **standard Factorization Machine**
    model = AFM(linear_feature_columns, dnn_feature_columns, task='binary', afm_dropout=dropout)
    model.compile(optimizer, "binary_crossentropy",
                  metrics=METRICS)

    log_dir = prefix_dir + 'afm_' + data_type + '_' + str(epochs)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[logs])
    pred_ans = model.predict(test_model_input, batch_size=256)

    output(history, test, pred_ans, target, 'afm', data_type, epochs, optimizer, dropout)
