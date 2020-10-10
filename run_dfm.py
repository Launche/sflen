import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    # data = pd.read_csv('./test.csv')
    data = pd.read_csv('/tmp/data/data_before_0510_smotenc.csv')
    test = pd.read_csv('/tmp/data/test.csv')

    # sparse_features = ['C' + str(i) for i in range(1, 27)]
    # dense_features = ['I' + str(i) for i in range(1, 14)]
    dense_features = ['price']

    sparse_features = ['cms_segid',
                       'cms_group_id',
                       'final_gender_code',
                       'age_level',
                       'pvalue_level',
                       'new_user_class_level',
                       'adgroup_id',
                       'pid',
                       'cate_id',
                       'campaign_id',
                       'customer',
                       'brand', ]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    test[sparse_features] = test[sparse_features].fillna('-1', )
    test[dense_features] = test[dense_features].fillna(0, )

    target = ['click']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        test[feat] = lbe.fit_transform(test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    test[dense_features] = mms.fit_transform(test[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train = data

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=0.4)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
