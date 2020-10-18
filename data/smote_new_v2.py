#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score  # 评估指标
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('/tmp/data/small_train.csv')

data.rename(columns={'hour': 'time'}, inplace=True)

data['time'] = data['time'].astype('str')
data['hour'] = data['time'].str[6:]

data.head()

columns = data.columns

sparse = ['C1', 'banner_pos', 'site_category', 'app_category',
          'device_type', 'device_conn_type', 'C18', 'C15', 'C16', 'hour']
dense = ['site_id', 'site_domain', 'app_id', 'app_domain', 'device_ip',
         'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']

for i in columns:
    print(data[i].value_counts())

column1 = ['C1', 'banner_pos', 'site_category', 'app_category',
           'device_type', 'device_conn_type', 'C18', 'C15', 'C16', 'hour',
           'site_id', 'site_domain', 'app_id', 'app_domain', 'device_ip',
           'device_model', 'C14', 'C17', 'C19', 'C20', 'C21']

list2 = []
for i in sparse:
    for num, j in enumerate(column1):
        # print(j)
        if j == i:
            list2.append(num)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

for i in column1:
    lbe = LabelEncoder()
    data[i] = lbe.fit_transform(data[i])
for j in dense:
    minmax = MinMaxScaler()
    data[j] = minmax.fit_transform(data[j].values.reshape(-1, 1))

data.head()

X = np.array(data[column1])
y = list(data['click'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[121]:


# smotenc
from imblearn.over_sampling import SMOTENC

smote_nc = SMOTENC(categorical_features=list2, random_state=0, sampling_strategy=0.3)
X_smotenc, y_smotenc = smote_nc.fit_resample(X_train, y_train)

# In[124]:


df_smotenc = pd.DataFrame(X_smotenc,
                          columns=column1)
df_smotenc = pd.concat([df_smotenc, pd.DataFrame(y_smotenc, columns=['click'])], axis=1)
for i in sparse:
    df_smotenc[i] = df_smotenc[i].astype(int)

# In[136]:

#
# # smotenc
# from imblearn.over_sampling import SMOTENC
#
# smote_nc1 = SMOTENC(categorical_features=list2, random_state=0, sampling_strategy=0.3)
# X_smotenc1, y_smotenc1 = smote_nc1.fit_resample(X_train, y_train)
#
# # In[138]:
#
#
# df_smotenc1 = pd.DataFrame(X_smotenc1,
#                            columns=column1)
# df_smotenc1 = pd.concat([df_smotenc1, pd.DataFrame(y_smotenc1, columns=['click'])], axis=1)
# for i in column1:
#     df_smotenc1[i] = df_smotenc1[i].astype(int)

df_smotenc.to_csv('/tmp/data/mayi_smotenc_train_03.csv', index=False)
print('work well done ...')
# # In[126]:
#
#
# df_smotenc.to_csv('/tmp/data/smotenc_new.csv', index=False)
#
# # In[131]:
#
#
# df_test = pd.DataFrame(X_test,
#                        columns=column1)
# df_test = pd.concat([df_test, pd.DataFrame(y_test, columns=['click'])], axis=1)
# for i in sparse:
#     df_test[i] = df_test[i].astype(int)
#
# # In[139]:
#
#
# df_train = pd.DataFrame(X_train,
#                         columns=column1)
# df_train = pd.concat([df_train, pd.DataFrame(y_train, columns=['click'])], axis=1)
# for i in sparse:
#     df_train[i] = df_train[i].astype(int)
# df_train.to_csv('/tmp/data/train_new.csv', index=False)
#
# # In[133]:
#
#
# df_test.to_csv('/tmp/data/test_new.csv', index=False)
#
# # In[135]:
#
#
# smotenc+enn
X_smote = np.array(df_smotenc[['C1', 'banner_pos', 'site_id', 'site_domain',
'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']])
Y_smote = list(df_smotenc['click'])
#
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_sample(X_smotenc, y_smotenc)
#
# # In[52]:
#
#
# df_smotenc = pd.DataFrame(X_smotenc,
#                           columns=column1)
# df_smotenc = pd.concat([df_smotenc, pd.DataFrame(y_smotenc, columns=['click'])], axis=1)
# for i in column1:
#     df_smotenc[i] = df_smotenc[i].astype(int)
#
# # In[53]:
#
#
# df_smX_resampledotenc.head()
#
# # In[56]:
#
#
# df_smotenc['click'].sum()
#
# # In[ ]:
#
#
# X_resampled
#
# # In[ ]:
