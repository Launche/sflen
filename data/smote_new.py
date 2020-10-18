#!/usr/bin/env python
# coding: utf-8
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pasta.augment import inline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours

data = pd.read_csv('/tmp/data/small_train.csv')
# data = data.drop(['id'], axis=1, inplace=True)

# data.head()


# trn = pd.read_csv('train.csv')
# target_col = trn.columns[-1]
# cat_cols = [col for col in trn.columns if trn[col].dtype == np.object]
#
# ohe = OneHotEncoder(min_obs=100) # grouping all categories with less than 100 occurences
# lbe = LabelEncoder(min_obs=100)  # grouping all categories with less than 100 occurences
# te = TargetEncoder()			       # replacing each category with the average target value of the category
# fe = FrequencyEncoder()			     # replacing each category with the frequency value of the category
# ee = EmbeddingEncoder()          # mapping each category to a vector of real numbers
#
# X_ohe = ohe.fit_transform(trn[cat_cols])	# X_ohe is a scipy sparse matrix
# trn[cat_cols] = lbe.fit_transform(trn[cat_cols])
# trn[cat_cols] = te.fit_transform(trn[cat_cols])
# trn[cat_cols] = fe.fit_transform(trn[cat_cols])
# X_ee = ee.fit_transform(trn[cat_cols])    # X_ee is a numpy matrix
#
# tst = pd.read_csv('test.csv')
# X_ohe = ohe.transform(tst[cat_cols])
# tst[cat_cols] = lbe.transform(tst[cat_cols])
# tst[cat_cols] = te.transform(tst[cat_cols])
# tst[cat_cols] = fe.transform(tst[cat_cols])
# X_ee = ee.transform(tst[cat_cols])

len_of_feature_count = []
for i in data.columns[2:].tolist():
    print(i, ':', len(data[i].astype(str).value_counts()))
    len_of_feature_count.append(len(data[i].astype(str).value_counts()))


# Get number of positve and negative examples
click = data[data["click"] == 1].shape[0]
unclick = data[data["click"] == 0].shape[0]
print("Good Applicant",click)
print("Bad Applicant",unclick)
plt.figure(figsize=(8, 6))
sns.countplot(data["click"])
plt.xticks((0, 1), ["Unclick", "Click"])
plt.xlabel("")
plt.ylabel("Count")
plt.title("Class counts", y=1, fontdict={"fontsize": 20})
plt.show()

# data['hour'] = data['hour'].astype('str')
# test = data[data['hour'].str[4:6] >= '29']
# train = data[data['hour'].str[4:6] < '29']


# data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
# data['hour'] = data['hour'].apply(lambda x: str(x)[6:])
#
# train, test = train_test_split(data, test_size=0.2)

#
# features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
#             'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
#             # 'device_ip',
#             'device_model', 'device_type', 'device_conn_type', 'C14',
#             'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
#
# categorical_features = ['hour', 'C1', 'banner_pos', 'site_domain',
#                         'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
#                         # 'device_ip',
#                         'device_model', 'device_type', 'device_conn_type', 'C14',
#                         'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
#
# data[features] = data[features].fillna('-1', )
#
# categorical_index = []
# for cf in categorical_features:
#     for num, feature in enumerate(features):
#         if cf == feature:
#             categorical_index.append(num)
#
# print(categorical_index)
#
# for feature in features:
#     lbe = LabelEncoder()
#     data[feature] = lbe.fit_transform(data[feature])
#
# X = np.array(data[['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
#                    'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',  # 'device_ip',
#                    'device_model', 'device_type', 'device_conn_type', 'C14',
#                    'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']])
#
# Y = list(data['click'])
#
# # smotenc
#
# # smote_nc = SMOTENC(categorical_features=categorical_index, random_state=0)
# smote_nc = SMOTENC(categorical_features=categorical_index, random_state=0, sampling_strategy=0.3)
# X_smotenc, y_smotenc = smote_nc.fit_resample(X, Y)
#
# df_smotenc = pd.DataFrame(X_smotenc,
#                           columns=features)
# df_smotenc = pd.concat([df_smotenc, pd.DataFrame(y_smotenc, columns=['click'])], axis=1)
#
# train = df_smotenc
#
# # test[feature] = lbe.fit_transform(test[feature])
#
# # # smotenc+enn
# # X_smote = np.array(df_smotenc[['C1', 'banner_pos', 'site_id', 'site_domain',
# #                                'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
# #                                'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
# #                                'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']])
# # Y_smote = list(df_smotenc['click'])
#
#
# # enn = EditedNearestNeighbours()
# # X_resampled, y_resampled = enn.fit_sample(X_smote, Y_smote)
#
#
# # for i in features:
# #     df_smotenc[i] = df_smotenc[i].astype(int)
#
#
# # Get number of positve and negative examples
# good = df_smotenc[df_smotenc["click"] == 1].shape[0]
# bad = df_smotenc[df_smotenc["click"] == 0].shape[0]
# print("Good Applicant", good)
# print("Bad Applicant", bad)
# plt.figure(figsize=(8, 6))
# sns.countplot(df_smotenc["click"])
# plt.xticks((0, 1), ["Bad Applicant", "Good Applicant"])
# plt.xlabel("")
# plt.ylabel("Count")
# plt.title("Class counts", y=1, fontdict={"fontsize": 20})
# plt.show()
#
# train.to_csv('./smote_v2.csv', index=False)
# # test.to_csv('./test.csv', index=False)
#
# print("write somete file")
# # df_smotenc['click'].sum()

# 1.先smote 再lbe (指定字段时，报错)
# 2.smotenc 不要指定字段
# 3.设置比率
# 4.drop 无用字段


#step1. 对数据集进行特征处理(比如lbe,归一化，生成连续特征),也可以删除无用特征
#step2. 随机切分训练与测试集 train_test_split() 并保存
#step3. 对训练集进行smotenc 代码：smote_nc = SMOTENC(categorical_features=categorical_index, random_state=0, sampling_strategy=0.3)
#step4. 保存训练集
