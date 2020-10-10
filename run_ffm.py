import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

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

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    # use_bn=False mean that it not use bn after ffm out
    model = ONN(linear_feature_columns, dnn_feature_columns, task='binary', use_bn=False)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=2, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
