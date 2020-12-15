import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from constant import sparse_features, dense_features
from database.db import insert
from plot_curves import history_curves


def get_data(origin, sampling_strategy=0.3):
    # if origin == "raw":
    #     data = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    #     train = data[data['day'] < 29]
    #     test = data[data['day'] >= 29]
    #     train.drop(columns=['id', 'day'], inplace=True)
    #     test.drop(columns=['id', 'day'], inplace=True)
    #     print("This model is using raw data ...")
    #
    #
    # elif origin == "enc":
    #     data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenc.csv')
    #     input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    #     train = data
    #     test = input_test[input_test['day'] >= 29]
    #     test.drop(columns=['id', 'day'], inplace=True)
    #     print("This model is using smote data ...")
    #
    # elif origin == "enn":
    #     data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenn.csv')
    #     input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    #     train = data
    #     test = input_test[input_test['day'] >= 29]
    #     test.drop(columns=['id', 'day'], inplace=True)
    #     print("This model is using smote + enn data ...")
    #
    # elif origin == "tlk":
    #     data = pd.read_csv('/tmp/data/avazu_data_100w_FE_tomeklink.csv')
    #     input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
    #     train = data
    #     test = input_test[input_test['day'] >= 29]
    #     test.drop(columns=['id', 'day'], inplace=True)
    #     print("This model is using tomeklink data ...")

    if origin == "enc":
        data = pd.read_csv('/tmp/data/mayi_smotenc_train_03.csv')
        test = pd.read_csv('/tmp/data/mayi_smotenc_test_03.csv')
        train = data
        print("=====================This model is using enc data ... =====================")

    elif origin == "raw":
        # data = pd.read_csv('/tmp/data/small_train.csv')
        data = pd.read_csv('/tmp/data/train.csv')
        data[sparse_features] = data[sparse_features].fillna('-1', )
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])

        for feat in dense_features:
            minmax = MinMaxScaler()
            data[feat] = minmax.fit_transform(data[feat].values.reshape(-1, 1))
        train, test = train_test_split(data, test_size=0.2, random_state=2020)
        print("=====================This model is using raw data ... =====================")

    elif origin == "extd":
        data = pd.read_csv('/tmp/data/all_raw_train.csv')
        test = pd.read_csv('/tmp/data/all_raw_test.csv')
        train = data
        print("=====================This model is using extd data ... =====================")

    elif origin == "part":
        data = pd.read_csv('/tmp/data/part_raw_train.csv')
        test = pd.read_csv('/tmp/data/part_raw_test.csv')
        train = data
        print("=====================This model is using part data ... =====================")

    elif origin == "enn":
        data = pd.read_csv('/tmp/data/mayi_smotenn_train_03.csv')
        test = pd.read_csv('/tmp/data/mayi_smotenc_test_03.csv')
        train = data
        print("=====================This model is using enn data ... =====================")

    elif origin == "tlk":
        data = pd.read_csv('/tmp/data/mayi_tomelink_train_03.csv')
        test = pd.read_csv('/tmp/data/mayi_smotenc_test_03.csv')
        train = data
        print("=====================This model is using tlk data ... =====================")

    elif origin == "batch":
        train_file_name = '/tmp/data/batch_smotenc_train_%s.csv' % (str(sampling_strategy).replace('.', ''))
        data = pd.read_csv(train_file_name)
        test = pd.read_csv('/tmp/data/batch_smotenc_test.csv')
        train = data
        print("=====================This model is using batch data: %s ... =====================" % train_file_name)
    return data, train, test


def output(history, test, pred_ans, target, algo, data_type, epoch, optimizer, dropout):
    logLoss = round(log_loss(test[target].values, pred_ans), 4)
    auc = round(roc_auc_score(test[target].values, pred_ans), 4)
    print("test LogLoss", logLoss)
    print("test AUC", auc)
    # history_curves(history)
    insert(algo, data_type, epoch, optimizer, dropout, logLoss, auc)
    print("Successfully insert ...")
