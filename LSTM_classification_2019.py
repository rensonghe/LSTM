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
data1 = pd.read_csv('test2103_01_04_01_08_depth.csv')
#%%
data = pd.read_csv('test2101_min_ru_withoutminfeature.csv')
#%%
data = pd.read_csv('test2108_2112_min.csv')
#%%
data = data.iloc[:,1:]
#%%
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['min'] = data['datetime'].dt.minute
data['sec'] = data['datetime'].dt.second
# data['day'] = data['datetime'].dt.day
#%%
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

data['sin_sec'] = sin_transformer(60).fit_transform(data['sec'])
data['cos_sec'] = cos_transformer(60).fit_transform(data['sec'])
data['sin_min'] = sin_transformer(60).fit_transform(data['min'])
data['cos_min'] = cos_transformer(60).fit_transform(data['min'])
data['sin_hour'] = sin_transformer(24).fit_transform(data['hour'])
data['cos_hour'] = cos_transformer(24).fit_transform(data['hour'])
# data = data.drop(['stime','datetime'],axis=1)
#%% separate time zone
# data['time'] = data['datetime'].str.extract('\s(.{0,15})')
import datetime
data['datetime'] = pd.to_datetime(data['datetime'])
start_time_night_1 = datetime.datetime.strptime('21:00:00.000','%H:%M:%S.%f').time()
end_time_night_1 = datetime.datetime.strptime('23:59:29.000','%H:%M:%S.%f').time()

start_time_night_2 = datetime.datetime.strptime('00:00:00.000','%H:%M:%S.%f').time()
end_time_night_2 = datetime.datetime.strptime('02:00:00.000','%H:%M:%S.%f').time()

data = data[(data.datetime.dt.time >= start_time_night_1) & (data.datetime.dt.time <= end_time_night_1)|
            (data.datetime.dt.time >= start_time_night_2) & (data.datetime.dt.time <= end_time_night_2)]
#%%
data = data[(data.datetime>='2022-01-04 09:00:00')& (data.datetime<='2022-01-08 08:59:59')]
#%%
data['target'] = np.log(data['close'] / data['close'].shift(1)) * 100
data['target'] = data['target'].shift(-1)
data = data.dropna(axis=0, how='any')
#%%
data = data.set_index('datetime_')
#%%
def classify(y):
    if y < 0:
        return -1
    elif y > 0:
        return 1
    else:
        return 0
data['target'] = data['target'].apply(lambda x:classify(x))
print(data['target'].value_counts())
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target

train_set = train[:39807]
test_set = train[39807:]
train_target = target[:39807]
test_target = target[39807:]

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

# from numpy import array
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

n_timestamp = 20

X_train, y_train, X_train_target, y_train_target = data_split(train_set_scaled,train_target, n_timestamp)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
X_test, y_test, X_test_target, y_test_target = data_split(test_set_scaled,test_target, n_timestamp)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target

# 将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train = sc.fit_transform(train)  # 数据归一
target = np.array(target)
target = to_categorical(target, num_classes=3)

train_X,test_X, train_y, test_y = train_test_split(train, target, test_size = 0.3, random_state = 10)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
#%%
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
              input_shape=(42, 1)))  # returns a sequence of vectors of dimension 30
model.add(Dropout(0.5))
model.add(LSTM(units=50)) # return a single vector of dimension 30
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()  # 输出模型结构

#%%
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#%%
n_epochs = 30
history = model.fit(train_X, train_y,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(test_X, test_y),  # revise
                    validation_freq=1)
#%%
n_epochs = 30
history = model.fit(y_train, y_train_target,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(y_test, y_test_target),  # revise
                    validation_freq=1)  # 测试的epoch间隔数
#%%
model.save('lstm_1.4_1.8_49var.h5')  # 保存
#%%
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
plt.show()   # 输出训练集测试集结果
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()   # 输出训练集测试集结果
#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy vs val_accuracy')
plt.ylabel('accuracy')
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




#%%
test_data = data.iloc[:,2:]
#%%
test_data = test_data[100000:150000]
test_data['target'] = np.log(test_data['wap1_mean'] / test_data['wap1_mean'].shift(1)) * 100
test_data['target'] = test_data['target'].shift(-1)
test_data = test_data.dropna(axis=0, how='any')
#%%
test_data = test_data[20:]
test_data = test_data.reset_index(drop=True)
#%%
predict = pd.DataFrame(result)
truth = pd.DataFrame(testY_label)
#%%
predict = predict[:49979]
#%%
solution = pd.concat([test_data, predict], axis=1)
#%%
solution.to_csv('solution_1tick.csv')