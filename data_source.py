import pandas as pd


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

    return data, train, test
