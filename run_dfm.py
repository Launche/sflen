import os
import sys

from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from tensorflow import optimizers

from constant import *
from utils import get_data, output

if __name__ == "__main__":
    for i in range(3, 11, 1):
        print(i / 10)
        # 1.prepare data and define epochs
        epochs = 30
        optimizer = "adam"
        dropout = 0
        data_type = 'batch'
        sampling_strategy = i / 10
        if sys.argv.__len__() == 4:
            data_type = sys.argv[1]
            epochs = int(sys.argv[2])
            sampling_strategy = float(sys.argv[3])

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        data, train, test = get_data(data_type, sampling_strategy)
        data_type = data_type + '_smotenc_' + str(sampling_strategy).replace('.', '')
        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                                  for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                                dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout)
        optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, "binary_crossentropy",
                      metrics=METRICS)

        log_dir = prefix_dir + 'dfm_' + data_type + '_' + str(epochs)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(train_model_input, train[target].values,
                            batch_size=256, epochs=epochs, verbose=2, validation_split=0.2,
                            callbacks=[logs])

        pred_ans = model.predict(test_model_input, batch_size=256)

        output(history, test, pred_ans, target, 'dfm', data_type, epochs, optimizer, dropout)
