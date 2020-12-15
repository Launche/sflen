import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from constant import sparse_features, dense_features

data = pd.read_csv('/tmp/data/train.csv')
data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
data['day'] = data['day'].astype('int')
# data['day'].nunique()
data = data[(data['day'] >= 24) & (data['day'] <= 28)]
# data = data[data['day']]
# for v in data['day'].unique():
#     print(v)

data[sparse_features] = data[sparse_features].fillna('-1', )
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

for feat in dense_features:
    minmax = MinMaxScaler()
    data[feat] = minmax.fit_transform(data[feat].values.reshape(-1, 1))

train, test = train_test_split(data, test_size=0.2, random_state=2020)

print('Starting write train file ....')
train.to_csv('/tmp/data/part_raw_train.csv')
print('Train file write successfully....')

print('Starting write test file ....')
test.to_csv('/tmp/data/part_raw_test.csv')
print('Test file write successfully....')
