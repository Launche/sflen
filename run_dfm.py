import pandas as pd
import sklearn
from matplotlib import colors
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import tensorflow as tf

from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import matplotlib.pyplot as plt

from RocAuc import RocAuc


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    # data = pd.read_csv('./test.csv')
    # input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    data = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    # data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenc.csv')
    # data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenn.csv')
    # test = pd.read_csv('/tmp/data/test.csv')

    train = data[data['day'] < 29]
    test = data[data['day'] >= 29]
    train.drop(columns=['id', 'day'], inplace=True)

    # train = data
    # test = input_test[input_test['day'] >= 29]

    test.drop(columns=['id', 'day'], inplace=True)

    sparse_features = ['C1', 'banner_pos', 'site_category', 'app_category',
                       'device_type', 'device_conn_type', 'C18', 'hour', 'is_device', 'C_pix']
    dense_features = ['C_site_id', 'C_site_domain', 'C_app_id', 'C_app_domain', 'C_device_ip',
                      'C_device_model', 'C_C14', 'C_C17', 'C_C19', 'C_C20', 'C_C21']
    target = ['click']
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

    METRICS = [
        # tf.keras.metrics.TruePositives(name='tp'),
        # tf.keras.metrics.FalsePositives(name='fp'),
        # tf.keras.metrics.TrueNegatives(name='tn'),
        # tf.keras.metrics.FalseNegatives(name='fn'),
        # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        # tf.keras.metrics.Precision(name='precision'),
        # tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        # tf.keras.metrics.Recall(name='recall'),
    ]

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=0.9)
    model.compile("adam", "binary_crossentropy",
                  metrics=METRICS)
    # model.summary()

    logs = tf.keras.callbacks.TensorBoard(log_dir='./test', histogram_freq=1)
    # logs = tf.keras.callbacks.TensorBoard(log_dir='./log/dfm_enc_log', histogram_freq=1)
    # logs = tf.keras.callbacks.TensorBoard(log_dir='./log/dfm_enn_log', histogram_freq=1)
    roc = RocAuc(data,test)
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=1, verbose=2, validation_split=0.2,
                        callbacks=[logs, roc])

    pred_ans = model.predict(test_model_input, batch_size=256)

    # print(pred_ans)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    #
    # train_labels = test[test['click'] == 1]
    # plot_roc("Train Baseline", train_labels, pred_ans, color='skyblue')
    # # plot_roc("Test Baseline", test_labels, test_predictions, color=colors[0], linestyle='--')
    # plt.legend(loc='lower right')
