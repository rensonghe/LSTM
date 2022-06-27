#%%
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
data = pd.read_csv('ni_2020.csv')
#%%
import ta
data_time = ta.utils.dropna(data)
data_time = ta.add_all_ta_features(data_time, "open", "high", "low", "close", "volume", fillna=True)
#%%
# civ_df = pd.read_csv('output_feature_detail_log_return.csv')
# # 删除iv值过小的变量
# iv_thre = 0.01
# iv = civ_df[['var_name', 'iv']].drop_duplicates()
# x_columns = iv.var_name[iv.iv > iv_thre]
# y_columns = x_columns.tolist()
# data = data_time.reindex(columns=y_columns)
#%%
data_filter = pd.read_csv('test.csv')
data_filter = data_filter.iloc[:,1:]
col = data_filter.columns
data = data_time.reindex(columns=col)
#%%
data['log_return'] = np.log(data['close'] / data['close'].shift(1)) * 100
data['log_return'] = data['log_return'].shift(-1)
data = data.dropna(axis=0, how='any')
#%%
data_time['log_return'] = np.log(data_time['close'] / data_time['close'].shift(1)) * 100
data_time['log_return'] = data_time['log_return'].shift(-1)
data_time = data_time.dropna(axis=0, how='any')
data_time = data_time.iloc[:,2:]
#%%
def classify(y):
    if y<-0.02:
        return -1
    elif y>0.02:
        return 1
    else:
        return 0
data_time['log_return'] = data_time['log_return'].apply(lambda x:classify(x))
#%%
cols = data_time.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "log_return":
        train_col.append(i)
#%%
train = data_time[train_col]
target = data_time["log_return"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[:80000]
test_set = train[-4000:]
train_target = target[:80000]
test_target = target[-4000:]
#%%
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
# train_X,test_X, train_y, test_y = train_test_split(train, target, test_size = 0.05, random_state = 0)
# #%%
# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
#%%
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
              input_shape=(90, 1)))  # returns a sequence of vectors of dimension 30
model.add(LSTM(units=50))  # return a single vector of dimension 30
# model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()  # 输出模型结构
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#%%
from tensorflow.keras.models import load_model
model = load_model('lstm_model_classification_26step_1min.h5')
#%%
n_epochs = 200
history = model.fit(y_train, y_train_target,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(y_test, y_test_target),  # revise
                    validation_freq=1)  # 测试的epoch间隔数
#%%
model.save('lstm_model_classification_2019+2020_26min_1min.h5')
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()   # 输出训练集测试集结果
#%%
predicted_contract_price = model.predict(y_test)  # 测试集输入模型进行预测

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
