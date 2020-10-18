from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from constant import sparse_features, dense_features, categorical_features
from imblearn.over_sampling import SMOTENC


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

    # 1.1 smotenc
    # categorical_features = ['hour', 'C1', 'banner_pos',
    #                         'site_category', 'app_category',  # 'device_ip',
    #                         'device_type', 'device_conn_type', 'C15', 'C16', 'C18']

    categorical_list = []
    for cf in categorical_features:
        for num, sf in enumerate(sparse_features):
            if cf == sf:
                categorical_list.append(num)

    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    X_train = pd.np.array(train[sparse_features])
    Y_train = list(train['click'])

    if sampling_strategy:
        print("This smotenc generater using " + str(sampling_strategy))
        smote_nc = SMOTENC(categorical_features=categorical_list, random_state=0, sampling_strategy=sampling_strategy)
    else:
        smote_nc = SMOTENC(categorical_features=categorical_list, random_state=0)

    X_smotenc, Y_smotenc = smote_nc.fit_resample(X_train, Y_train)

    train = pd.DataFrame(X_smotenc,
                              columns=sparse_features)
    train = pd.concat([train, pd.DataFrame(Y_smotenc, columns=['click'])], axis=1)

    for i in categorical_features:
        train[i] = train[i].astype(int)

    print("writing trian file ...")
    train.to_csv('/tmp/data/mayi_smotenc_train_03.csv', index=False)
    print("trian file write done")

    print("writing test file ...")
    test.to_csv('/tmp/data/mayi_smotenc_test_03.csv', index=False)
    print("test file write done")


if __name__ == "__main__":
    smotenc_generater(0.3)
