import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import gzip

from sklearn.model_selection import train_test_split

data = pd.read_csv('/tmp/data/small_train.csv')


# 建立一個將hour資料轉換為日期格式的function
def get_date(hour):
    y = '20' + str(hour)[:2]
    m = str(hour)[2:4]
    d = str(hour)[4:6]
    return y + '-' + m + '-' + d


# 建立weekday欄位，將hour轉換後填入
data['weekday'] = pd.to_datetime(data.hour.apply(get_date)).dt.dayofweek.astype(str)

87

# 建立一個將hour資料轉換為時段的function
def tran_hour(x):
    x = x % 100
    while x in [23, 0]:
        return '23-01'
    while x in [1, 2]:
        return '01-03'
    while x in [3, 4]:
        return '03-05'
    while x in [5, 6]:
        return '05-07'
    while x in [7, 8]:
        return '07-09'
    while x in [9, 10]:
        return '09-11'
    while x in [11, 12]:
        return '11-13'
    while x in [13, 14]:
        return '13-15'
    while x in [15, 16]:
        return '15-17'
    while x in [17, 18]:
        return '17-19'
    while x in [19, 20]:
        return '19-21'
    while x in [21, 22]:
        return '21-23'


# 將hour轉換為時段
data['hour'] = data.hour.apply(tran_hour)

# 统计特征类数
len_of_feature_count = []
for i in data.columns[2:].tolist():
    print(i, ':', len(data[i].astype(str).value_counts()))
    len_of_feature_count.append(len(data[i].astype(str).value_counts()))

# 建立一個list，將需要轉換行別的特徵名稱存入該list
need_tran_feature = data.columns[2:4].tolist() + data.columns[13:23].tolist()

# 依序將變數轉換為object型別
for i in need_tran_feature:
    data[i] = data[i].astype(str)

obj_features = []

for i in range(len(len_of_feature_count)):
    if len_of_feature_count[i] > 10:
        obj_features.append(data.columns[2:].tolist()[i])
print(obj_features)

df_describe = data.describe()


def obj_clean(X):
    # 定義一個縮減資料值的function，每次處理一個特徵向量

    def get_click_rate(x):
        # 定義一個取得點擊率的function
        temp = train[train[X.columns[0]] == x]
        res = round((temp.click.sum() / temp.click.count()), 3)
        return res

    def get_type(V, str):
        # 定義一個取得新資料值之級距判斷的function
        very_high = df_describe.loc['mean', 'click'] + 0.04
        higher = df_describe.loc['mean', 'click'] + 0.02
        lower = df_describe.loc['mean', 'click'] - 0.02
        very_low = df_describe.loc['mean', 'click'] - 0.04

        vh_type = V[V[str] > very_high].index.tolist()
        hr_type = V[(V[str] > higher) & (V[str] < very_high)].index.tolist()
        vl_type = V[V[str] < very_low].index.tolist()
        lr_type = V[(V[str] < lower) & (V[str] > very_low)].index.tolist()

        return vh_type, hr_type, vl_type, lr_type

    def clean_function(x):
        # 定義一個依據級距轉換資料值的function
        # 判斷之依據為：總平均點擊率的正負  4% 為very_high(low), 總平均點擊率的正負 2％為higher (lower)
        while x in type_[0]:
            return 'very_high'
        while x in type_[1]:
            return 'higher'
        while x in type_[2]:
            return 'very_low'
        while x in type_[3]:
            return 'lower'
        return 'mid'

    print('Run: ', X.columns[0])
    fq = X[X.columns[0]].value_counts()
    # 建立一個暫存的資料值頻率列表
    # 理論上，將全部的資料值都進行分類轉換，可得到最佳效果；實務上為了執行時間效能，將捨去頻率低於排名前1000 row以後的資料值。
    if len(fq) > 1000:
        fq = fq[:1000]

    # 將頻率列表轉換為dataframe，並將index填入一個新的欄位。
    fq = pd.DataFrame(fq)
    fq['new_column'] = fq.index

    # 使用index叫用get_click_rate function，取得每個資料值的點擊率
    fq['click_rate'] = fq.new_column.apply(get_click_rate)

    # 叫用 get_type function取得分類級距，並儲存為一個list，以便提供給下一個clean_function使用
    type_ = get_type(fq, 'click_rate')

    # 叫用 clean_funtion funtion，回傳轉換後的特徵向量
    return X[X.columns[0]].apply(clean_function)


# 使用for 迴圈將需轉換的特徵輸入到 obj_clean function
list_features=['device_id','device_ip','device_model','site_domain','site_id','app_id','app_id']
for i in list_features:
    data[[i]] = obj_clean(data[[i]])

for i in data.columns:
    sns.countplot(x=i, hue="click", data=data)
    plt.show()
