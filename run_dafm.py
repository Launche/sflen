import os
import sys
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from constant import *
from utils import get_data, output
from models.deepfm import DeepFM2

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 100
    optimizer = "adam"
    dnn_dropout = 0.5
    afm_dropout = 0.6
    dropout = 'dnn_dropout: ' + str(dnn_dropout) + ';afm_dropout: ' + str(afm_dropout)

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

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM2(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dnn_dropout,
                    afm_dropout=afm_dropout)

    # opt = tf.keras.optimizers.SGD
    model.compile(optimizer, "binary_crossentropy",
                  metrics=METRICS)
    # model.summary()

    log_dir = prefix_dir + 'dafm_' + data_type + '_' + str(epochs)
    if not os.path.exists(log_dir):  # 如果路径不存在
        os.makedirs(log_dir)

    logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=epochs, verbose=2, validation_split=0.2,
                        callbacks=[logs])

    pred_ans = model.predict(test_model_input, batch_size=256, callbacks=[logs])

    output(history, test, pred_ans, target, 'dafm', data_type, epochs, optimizer, dropout)
