import os
import sys

from deepctr.models import AutoInt
from tensorflow.python.debug.examples.v1.debug_keras import tf
from deepctr.feature_column import SparseFeat, get_feature_names, DenseFeat
from constant import sparse_features, METRICS, prefix_dir, target, dense_features
from utils import output, get_data

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 10
    optimizer = "adam"
    dropout = 0
    data_type = 'part'
    if sys.argv.__len__() == 3:
        data_type = sys.argv[1]
        epochs = int(sys.argv[2])

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
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
    model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout)
    model.compile(optimizer, "binary_crossentropy",
                  metrics=METRICS)

    log_dir = prefix_dir + 'autoint_' + data_type + '_' + str(epochs)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[logs])
    pred_ans = model.predict(test_model_input, batch_size=256)

    output(history, test, pred_ans, target, 'autoint', data_type, epochs, optimizer, dropout)
