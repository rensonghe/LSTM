import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import tensorflow as tf
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


#%%
data = pd.read_csv('test_2021.csv')
#%%
data = data.set_index(['time'])
#%%
test_set = data[(data.index>='2021-01-01 0:00:00')& (data.index<='2021-12-31 23:59:59')]
#%%
import ta
data_time = ta.utils.dropna(test_set)
data_time = ta.add_all_ta_features(data_time, "open", "high", "low", "close", "volume", fillna=True)
#%%
data_filter = pd.read_csv('test.csv')
data_filter = data_filter.iloc[:,1:]
col = data_filter.columns
#%%
data = data_time.reindex(columns=col)
#%%
data['log_return'] = np.log(data['close'] / data['close'].shift(1)) * 100
data['log_return'] = data['log_return'].shift(1)
data = data.dropna(axis=0, how='any')
#%%
def classify(y):
    if y<-0.02:
        return -1
    elif y>0.02:
        return 1
    else:
        return 0
data['log_return'] = data['log_return'].apply(lambda x:classify(x))
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "log_return":
        train_col.append(i)
#%%
train = data[train_col]
target = data["log_return"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[:55584]
test_set = train[-55584:]
train_target = target[:55584]
test_target = target[-55584:]

# 将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)# 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)

from keras.utils.np_utils import to_categorical
train_target = to_categorical(train_target, num_classes=3)
test_target = to_categorical(test_target, num_classes=3)
#%%
# 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
def data_split(sequence,target ,n_timestamp):
    X = []
    y = []
    X_target = []
    y_target = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        seq_target_x, seq_target_y = target[i:end_ix], target[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        X_target.append(seq_target_x)
        y_target.append(seq_target_y)

    return array(X), array(y), array(X_target), array(y_target)

n_timestamp = 25

X_train, y_train, X_train_target, y_train_target = data_split(train_set_scaled,train_target, n_timestamp)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
X_test, y_test, X_test_target, y_test_target = data_split(test_set_scaled,test_target, n_timestamp)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
#%%
# 将数据归一化，范围是0到1
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
# train = sc.fit_transform(train)  # 数据归一
# target = np.array(target)
#
# #%%
# from keras.utils.np_utils import to_categorical
# target = to_categorical(target, num_classes=3)
#
# #%%
# # 划分训练集测试集
# from sklearn.model_selection import train_test_split
# train_X,test_X, train_y, test_y = train_test_split(train, target, test_size = 0.5, random_state = 1)
# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

#%%
from tensorflow.keras.models import load_model
model = load_model('lstm_model_classification_2019+2020_26min_1min.h5')
#%%
n_epochs = 50
history = model.fit(y_train, y_train_target,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(y_test, y_test_target),  # revise
                    validation_freq=1)  # 测试的epoch间隔数

#%%
predicted_contract_price = model.predict(y_test)
#%%
# 将预测结果转为标签形式
def to_label(result):
    label = []
    for i in result:
        l = np.argmax(i)
        if l == 0:
            label.append(0)
        elif l == 1:
            label.append(1)
        else:
            label.append(-1)
    return label
#%%
result = to_label(predicted_contract_price)  # 测试集预测类别结果
testY_label = to_label(y_test_target)

#%%
from sklearn.metrics import classification_report
print("测试集表现：")
print(classification_report(testY_label, result))
#%%
from sklearn.metrics import accuracy_score
from sklearn import metrics
print("评价指标-测试集精确率score：")
print(accuracy_score(testY_label, result))

print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(testY_label, result))
#%%
predict_result = pd.DataFrame(result)
#%%
true_result = pd.DataFrame(testY_label)
col_1 = ['true']
true_result.columns = col_1
#%%
col = ['predict']
predict_result.columns = col
#%%
c = true_result.join(predict_result)
#%%
test_set = test_set.reset_index()
#%%
final_result = test_set[-55559:]
#%%
final_result = final_result.reset_index(drop=True)
#%%
last_result = final_result.join(predict_result)
#%%
# col_1= ['time', 'close', 'predict']
# test_result = final_result.loc[:,['time', 'close', 'predict']]
#%%
last_result.to_csv('test_result_classification_2021_26min_1min.csv')
#%%
b = len(c[(c.true>0)&(c.predict>0)|((c.true<0)&(c.predict<0))])/len(c)
#%%
def inverse_transform_col(scaler, y, n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= sc.min_[n_col]
    y /= sc.scale_[n_col]
    return y

#%%
# 第3列归一化的值
predict = y_test[:,0,:]
#%%
# 第3列反归一化
predict_col_0 = inverse_transform_col(sc, predict, n_col=0)

#%%
# 真实值返归一化
true = y_test[:,0,:]
true_col_0 = inverse_transform_col(sc, true, n_col=0)
#%%
truth = true_col_0[:, -1]
forecast = predict_col_0[:,-1]
#%%
a = pd.read_csv('test_result_classification_2021_26min_1min.csv')
a['predict'] = a['predict'].shift(1)
#%%
a.to_csv('a.csv')