import os
import sys

import pandas as pd
from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


from constant import *
from utils import  output

# def plot_roc(name, labels, predictions, **kwargs):
#     fp, tp, _ = roc_curve(labels, predictions)
#     plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
#     plt.xlabel('False positives [%]')
#     plt.ylabel('True positives [%]')
#     plt.xlim([-0.5, 20])
#     plt.ylim([80, 100.5])
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')
#

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 10
    optimizer = "adam"
    dropout = 0
    data_type = 'enc'
    if sys.argv.__len__() == 3:
        data_type = sys.argv[1]
        epochs = int(sys.argv[2])

    # data, train, test = get_data(data_type)
    data = pd.read_csv('/tmp/data/mayi_smotenc_train_03.csv')
    train = data
    test = pd.read_csv('/tmp/data/mayi_smotenc_test_03.csv')
    # data = pd.read_csv('/tmp/data/small_train.csv')

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # test[sparse_features] = test[sparse_features].fillna('-1', )
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(test[feat])
    #
    # for feat in dense_features_test:
    #     minmax = MinMaxScaler()
    #     test[feat] = minmax.fit_transform(test[feat].values.reshape(-1, 1))

    # # 1.1 smotenc
    # categorical_features = ['hour', 'C1', 'banner_pos',
    #                         'site_category', 'app_category',  # 'device_ip',
    #                         'device_type', 'device_conn_type', 'C15', 'C16', 'C18']
    # categorical_list = []
    # for cf in categorical_features:
    #     for num, sf in enumerate(sparse_features):
    #         if cf == sf:
    #             categorical_list.append(num)
    #
    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    #
    # X_train = pd.np.array(train[sparse_features])
    # Y_train = list(train['click'])
    #
    # from imblearn.over_sampling import SMOTENC
    #
    # # smote_nc = SMOTENC(categorical_features=categorical_list, random_state=0, sampling_strategy=0.3)
    # smote_nc = SMOTENC(categorical_features=categorical_list, random_state=0)
    # X_smotenc, y_smotenc = smote_nc.fit_resample(X_train, Y_train)
    #
    # train = pd.DataFrame(X_train,
    #                      columns=sparse_features)
    # train = pd.concat([train, pd.DataFrame(Y_train, columns=['click'])], axis=1)

    # for i in categorical_features:
    #     train[i] = train[i].astype(int)

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                            dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout)

    # opt = tf.keras.optimizers.SGD
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
