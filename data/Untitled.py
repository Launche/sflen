#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import calendar
from datetime import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# In[2]:


types_train = {
    'id': np.dtype(int),
    'click': np.dtype(int),  # 是否点击,1表示被点击,0表示没被点击
    'hour': np.dtype(int),  # 广告被展现的日期+时间
    'C1': np.dtype(int),  # 匿名分类变量
    'banner_pos': np.dtype(int),  # 广告位置
    'site_id': np.dtype(str),  # 站点Id
    'site_domain': np.dtype(str),  # 站点域名
    'site_category': np.dtype(str),  # 站点分类
    'app_id': np.dtype(str),  # appId
    'app_domain': np.dtype(str),  # app域名
    'app_category': np.dtype(str),  # app分类
    'device_id': np.dtype(str),  # 设备Id
    'device_ip': np.dtype(str),  # 设备Ip
    'device_model': np.dtype(str),  # 设备型号
    'device_type': np.dtype(int),  # 设备型号
    'device_conn_type': np.dtype(int),
    'C14': np.dtype(int),  # 匿名分类变量
    'C15': np.dtype(int),  # 匿名分类变量
    'C16': np.dtype(int),  # 匿名分类变量
    'C17': np.dtype(int),  # 匿名分类变量
    'C18': np.dtype(int),  # 匿名分类变量
    'C19': np.dtype(int),  # 匿名分类变量
    'C20': np.dtype(int),  # 匿名分类变量
    'C21': np.dtype(int)  # 匿名分类变量
}

# In[3]:


n = 40428967  # 数据集中的记录总数
sample_size = 1000000
skip_values = sorted(random.sample(range(1, n), n - sample_size))
# parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')

with open('/Users/ljy/Desktop/paper/data/avazu-ctr-prediction/train.csv') as f:
    # train = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date, dtype=types_train, skiprows = skip_valus)
    train = pd.read_csv(f, skiprows=skip_values)
print(len(train))
train.head()

train.dtypes

train.columns.to_list()

for i in ['C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']:
    print(i, train[i].value_counts())

n_test = 4577464
sample_size_test = 100000
skip_values_test = sorted(random.sample(range(1, n_test), n_test - sample_size_test))
with open('/Users/ljy/Desktop/paper/data/avazu-ctr-prediction/test') as f:
    test = pd.read_csv(f, skiprows=skip_values_test)
print(len(test))
test.head()



train = pd.read_csv('./data/small_train.csv')


print(train.columns)

print('train', train['click'].value_counts())
print('ctr', train['click'].sum() / train['click'].count())

train['time'] = train.hour % 100
train.groupby('time').agg({'click': 'sum'}).plot(figsize=(12, 6), grid=True)
plt.ylabel('点击量')
plt.xlabel('时段')
plt.title('时段与点击量关系')

# train['day_of_week'] = train['hour'].apply(lambda val: val.weekday_name)
train['day'] = np.round(train.hour % 10000 / 100)
train['day'].value_counts()

print(train.columns)

train['C18'].value_counts()


train['device_id'].value_counts()

tr_ts = train.copy(False)

tr_ts['is_device'] = tr_ts['device_id'].apply(lambda x: 0 if x == 'a99f214a' else 1)
tr_ts.drop(columns=['device_id'], inplace=True)


tr_ts["C_pix"] = tr_ts["C15"].astype('str') + '&' + tr_ts["C16"].astype('str')
tr_ts.drop(columns=['C15', 'C16'], inplace=True)


label = ['C1', 'banner_pos', 'device_type', 'site_category', 'app_category', 'device_type', 'is_device',
         'device_conn_type', 'C_pix', 'C18']
sparse = ['site_id', 'site_domain', 'app_id', 'app_domain', 'device_ip', 'device_model'
    , 'C14', 'C17', 'C19', 'C20', 'C21']


# 计数编码
def vert_feature(feature_name, dataframe):
    data = dataframe.groupby(feature_name, as_index=False).agg({'click': 'count'})
    # print(data)
    label2data = dict(zip(data[feature_name], data['click']))
    # print(label2data)
    # data2label = dict(zip(sorted(list(set(train['site_id']))),range(0,len(set(train['site_id'])))))
    new_feature_name = 'C_' + feature_name
    dataframe[new_feature_name] = dataframe[feature_name].map(label2data)  # 编码
    # print(dataframe[new_feature_name].value_counts())
    return dataframe


for i in sparse:
    vert_feature(i, tr_ts)
tr_ts.drop(columns=sparse, inplace=True)


# 标签编码
lenc = preprocessing.LabelEncoder()
for f, column in enumerate(label):
    print("convert " + column + "...")
    tr_ts[column] = lenc.fit_transform(tr_ts[column])

# 数据归一化
c_sparse = []
for i in sparse:
    c_sparse.append('C_' + i)
mini_max_scaler = preprocessing.MinMaxScaler()
for i, column in enumerate(c_sparse):
    print("convert " + column + "...")
    tr_ts[column] = mini_max_scaler.fit_transform(tr_ts[column].values.reshape(-1, 1))

tr_ts['day'] = tr_ts['day'].astype('int')


tr_ts.drop(columns=['hour'], inplace=True)


tr_ts.rename(columns={'time': 'hour'}, inplace=True)


tr_ts.to_csv('./data/avazu_data_100w_FE.csv', index=False)

print(tr_ts.columns)


tr_ts['C_C20'].value_counts()

# ### 特征筛选

####对columns1 进行 one-hot编码
label1 = ['C1', 'hour', 'banner_pos', 'device_type', 'site_category', 'app_category', 'device_type', 'is_device',
          'device_conn_type', 'C_pix', 'C18']
train_concat = tr_ts.copy(False)
for i, column in enumerate(label1):
    new_column = pd.get_dummies(train_concat[column], prefix=column)
    train_concat = pd.concat([train_concat, new_column], axis=1)

# In[192]:


train_concat.drop(columns=label1, inplace=True)

# In[193]:


train_concat.head()

# In[176]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score  # 评估指标
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# In[ ]:
