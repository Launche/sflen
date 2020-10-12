import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf

from deepctr.feature_column import SparseFeat, get_feature_names
from deepctr.models import FLEN

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

    field_info = dict(C_C14='user', C_C17='user',
                      C18='user', C_C19='user', C_C20='user', C_C21='user', C1='user',
                      banner_pos='context', C_site_id='context',
                      C_site_domain='context', site_category='context',
                      C_app_id='item', C_app_domain='item', app_category='item',
                      C_device_model='user', device_type='user',
                      device_conn_type='context', hour='context',
                      C_device_ip='user',
                      is_device='user',
                      C_pix='user'
                      )

    fixlen_feature_columns = [
        SparseFeat(name, vocabulary_size=data[name].nunique(), embedding_dim=16, use_hash=False, dtype='int32',
                   group_name=field_info[name]) for name in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    # train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        # tf.keras.metrics.Precision(name='precision'),
        # tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]


    # 4.Define Model,train,predict and evaluate
    model = FLEN(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=0.8)
    model.compile("adam", "binary_crossentropy",
                  metrics=METRICS)

    logs = tf.keras.callbacks.TensorBoard(log_dir='./log/flen_raw_log', histogram_freq=1)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=500, verbose=2, validation_split=0.2, callbacks=[logs])
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
