import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from database.db import insert
from plot_curves import history_curves


def get_data(origin):
    if origin == "raw":
        data = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
        train = data[data['day'] < 29]
        test = data[data['day'] >= 29]
        train.drop(columns=['id', 'day'], inplace=True)
        test.drop(columns=['id', 'day'], inplace=True)
        print("This model is using raw data ...")


    elif origin == "enc":
        data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenc.csv')
        input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
        train = data
        test = input_test[input_test['day'] >= 29]
        test.drop(columns=['id', 'day'], inplace=True)
        print("This model is using smote data ...")

    elif origin == "enn":
        data = pd.read_csv('/tmp/data/avazu_data_100w_FE_smotenn.csv')
        input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
        train = data
        test = input_test[input_test['day'] >= 29]
        test.drop(columns=['id', 'day'], inplace=True)
        print("This model is using smote + enn data ...")

    elif origin == "tlk":
        data = pd.read_csv('/tmp/data/avazu_data_100w_FE_tomeklink.csv')
        input_test = pd.read_csv('/tmp/data/avazu_data_100w_FE.csv')
        train = data
        test = input_test[input_test['day'] >= 29]
        test.drop(columns=['id', 'day'], inplace=True)
        print("This model is using tomeklink data ...")

    return data, train, test


def output(history, test, pred_ans, target, algo, data_type, epoch, optimizer, dropout):
    logLoss = round(log_loss(test[target].values, pred_ans), 4)
    auc = round(roc_auc_score(test[target].values, pred_ans), 4)
    print("test LogLoss", logLoss)
    print("test AUC", auc)
    history_curves(history)
    insert(algo, data_type, epoch, optimizer, dropout, logLoss, auc)
