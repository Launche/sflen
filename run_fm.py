import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn import metrics
from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    # data = pd.read_csv('./test.csv')
    input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    # data = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenc.csv')
    # data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenn.csv')
    # test = pd.read_csv('/tmp/data/test.csv')

    train = data
    test = input_test[input_test['day'] >= 29]
    # train.drop(columns=['id', 'day'], inplace=True)
    test.drop(columns=['id', 'day'], inplace=True)

    # sparse_features = ['C' + str(i) for i in range(1, 27)]
    # dense_features = ['I' + str(i) for i in range(1, 14)]

    sparse_features = ['C1', 'banner_pos', 'site_category', 'app_category',
                       'device_type', 'device_conn_type', 'C18', 'hour', 'is_device', 'C_pix']
    dense_features = ['C_site_id', 'C_site_domain', 'C_app_id', 'C_app_domain', 'C_device_ip',
                      'C_device_model', 'C_C14', 'C_C17', 'C_C19', 'C_C20', 'C_C21']

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    #
    # test[sparse_features] = test[sparse_features].fillna('-1', )
    # test[dense_features] = test[dense_features].fillna(0, )

    target = ['click']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    #     test[feat] = lbe.fit_transform(test[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])
    # test[dense_features] = mms.fit_transform(test[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    # fixlen_feature_columns = [SparseFeat(feat, våocabulary_size=data[feat].nunique(), embedding_dim=4)
    #                           for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
    #                                                                         for feat in dense_features]
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    # use_attention=False mean that it is the same as **standard Factorization Machine**
    model = AFM(linear_feature_columns, dnn_feature_columns, task='binary', use_attention=False)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))



    # fpr_0, tpr_0, threshold_0 = metrics.roc_curve(target, pred_ans)
    # # fpr_1, tpr_1, threshold_1 = metrics.roc_curve(target, pred_ans)
    # # fpr_2, tpr_2, threshold_2 = metrics.roc_curve(target, pred_ans)
    # roc_auc_0 = metrics.auc(fpr_0, tpr_0)
    # # roc_auc_1 = metrics.auc(fpr_1, tpr_1)
    # # roc_auc_2 = metrics.auc(fpr_2, tpr_2)
    # plt.figure(figsize=(6, 6))
    # plt.title('Validation ROC')
    # plt.plot(fpr_0, tpr_0, 'b', label='Val AUC = %0.3f' % roc_auc_0)
    # # plt.plot(fpr_1, tpr_1, 'g', label='Val AUC = %0.3f' % roc_auc_1)
    # # plt.plot(fpr_2, tpr_2, 'orange', label='Val AUC = %0.3f' % roc_auc_2)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()
