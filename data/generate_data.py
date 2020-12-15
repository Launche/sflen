import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from constant import sparse_features, dense_features
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import seaborn as sns


categorical_features = ['hour', 'C1', 'banner_pos',
                        'site_category', 'app_category',
                        'device_type', 'device_conn_type', 'C15', 'C16', 'C18']


def smotenc_generater(sampling_strategy=None):
    data = pd.read_csv('/tmp/data/small_train.csv')
    data.rename(columns={'hour': 'time'}, inplace=True)
    data['time'] = data['time'].astype('str')
    data['hour'] = data['time'].str[6:]

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    data[sparse_features] = data[sparse_features].fillna('-1', )
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    for feat in dense_features:
        minmax = MinMaxScaler()
        data[feat] = minmax.fit_transform(data[feat].values.reshape(-1, 1))

    categorical_list = []
    for cf in categorical_features:
        for num, sf in enumerate(sparse_features):
            if cf == sf:
                categorical_list.append(num)

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    # X_train = pd.np.array(train[sparse_features])
    # Y_train = list(train['click'])

    X = pd.np.array(data[sparse_features])
    Y = list(data['click'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    raw = pd.DataFrame(X_train,
                         columns=sparse_features)
    raw = pd.concat([raw, pd.DataFrame(Y_train, columns=['click'])], axis=1)

    good = raw[raw["click"] == 1].shape[0]
    bad = raw[raw["click"] == 0].shape[0]
    print("Good Applicant", good)
    print("Bad Applicant", bad)
    plt.figure(figsize=(8, 6))
    sns.countplot(raw["click"])
    plt.xticks((0, 1), ["Bad Applicant", "Good Applicant"])
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title("Raw counts", y=1, fontdict={"fontsize": 20})
    plt.show()

    if sampling_strategy:
        print("This smotenc generater using " + str(sampling_strategy))
        smote_nc = SMOTENC(categorical_features=categorical_list, random_state=0, sampling_strategy=sampling_strategy)
    else:
        smote_nc = SMOTENC(categorical_features=categorical_list, random_state=0)

    X_smotenc, Y_smotenc = smote_nc.fit_resample(X_train, Y_train)

    train = pd.DataFrame(X_smotenc,
                         columns=sparse_features)
    train = pd.concat([train, pd.DataFrame(Y_smotenc, columns=['click'])], axis=1)
    test = pd.DataFrame(X_test,
                        columns=sparse_features)
    test = pd.concat([test, pd.DataFrame(Y_test, columns=['click'])], axis=1)

    for i in categorical_features:
        train[i] = train[i].astype(int)
        test[i] = test[i].astype(int)

    print("writing trian file ...")
    train_file_name = '/tmp/data/batch_smotenc_train_%s.csv' % (str(sampling_strategy).replace('.', ''))
    train.to_csv(train_file_name, index=False)
    print("trian file  %s write done" % train_file_name)

    # print("writing test file ...")
    # test.to_csv('/tmp/data/batch_smotenc_test.csv', index=False)
    # print("test file write done")

    good = train[train["click"] == 1].shape[0]
    bad = train[train["click"] == 0].shape[0]
    print("Good Applicant", good)
    print("Bad Applicant", bad)
    plt.figure(figsize=(8, 6))
    sns.countplot(train["click"])
    plt.xticks((0, 1), ["Bad Applicant", "Good Applicant"])
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title("Smotenc counts", y=1, fontdict={"fontsize": 20})
    plt.show()

if __name__ == "__main__":
    # split_test_train()
    sampling_strategy = 0.8
    if sys.argv.__len__() == 2:
        sampling_strategy = float(sys.argv[1])
    smotenc_generater(sampling_strategy)
