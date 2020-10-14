import os
import sys


from deepctr.feature_column import SparseFeat, get_feature_names
from deepctr.models import FLEN
from constant import *
from utils import get_data, output

if __name__ == "__main__":

    # 1.prepare data and define epochs
    epochs = 100
    optimizer = "adam"
    dropout = 0.5

    if sys.argv.__len__() == 3:
        data_type = sys.argv[1]
        epochs = int(sys.argv[2])

    data, train, test = get_data(data_type)

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
