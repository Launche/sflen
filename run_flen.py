import os
import sys
import pandas as pd

from deepctr.feature_column import SparseFeat, get_feature_names
from deepctr.models import FLEN
from sklearn.model_selection import train_test_split

from constant import *
from utils import get_data, output

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 10
    optimizer = "adam"
    dropout = 0
    data_type = "test"

    if sys.argv.__len__() == 3:
        data_type = sys.argv[1]
        epochs = int(sys.argv[2])

    # data, train, test = get_data(data_type)

    # data = pd.read_csv('/tmp/data/all_new.csv')
    # data = pd.read_csv('/tmp/data/all_new.csv')
    # data = pd.read_csv('./data/smote_v1.csv')
    data = pd.read_csv('/tmp/data/smotenc_new.csv')
    test = pd.read_csv('/tmp/data/test_new.csv')

    # 2.count #unique features for each sparse field,and record dense feature field name

    # field_info = dict(C_C14='user', C_C17='user',
    #                   C18='user', C_C19='user', C_C20='user', C_C21='user', C1='user',
    #                   banner_pos='context', C_site_id='context',
    #                   C_site_domain='context', site_category='context',
    #                   C_app_id='item', C_app_domain='item', app_category='item',
    #                   C_device_model='user', device_type='user',
    #                   device_conn_type='context', hour='context',
    #                   C_device_ip='user',
    #                   is_device='user',
    #                   C_pix='user'
    #                   )
    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category',
                       # 'device_id',
                       'device_model', 'device_type', 'device_conn_type', #'device_ip',
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]

    field_info = dict(C14='user', C15='user', C16='user', C17='user',
                      C18='user', C19='user', C20='user', C21='user', C1='user',
                      banner_pos='context', site_id='context',
                      site_domain='context', site_category='context',
                      app_id='item', app_domain='item', app_category='item',
                      device_model='user', device_type='user',
                      device_conn_type='context', hour='context'
                      # device_id='user'
                      )
    fixlen_feature_columns = [
        SparseFeat(name, vocabulary_size=data[name].nunique(), embedding_dim=16, use_hash=False, dtype='int32',
                   group_name=field_info[name]) for name in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    # train, test = train_test_split(data, test_size=0.2)
    train = data
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = FLEN(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout, dnn_use_bn=True)
    model.compile(optimizer, "binary_crossentropy",
                  metrics=METRICS)

    log_dir = prefix_dir + 'flen_' + data_type + '_' + str(epochs)
    if not os.path.exists(log_dir):  # 如果路径不存在
        os.makedirs(log_dir)

    logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[logs])
    pred_ans = model.predict(test_model_input, batch_size=256)
    output(history, test, pred_ans, target, 'flen', data_type, epochs, optimizer, dropout)
