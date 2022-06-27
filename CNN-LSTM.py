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
print(tf.test.gpu_device_name())
#%%
data = pd.read_csv('test.csv')
#%%
data['log_return'] = np.log(data['close'] / data['close'].shift(1))
# data['log_return'][np.isinf(data['log_return'])] = 0
data = data.dropna(axis=0, how='any')
#%%
data = data.iloc[:,1:]
#%%
# def evaluate_forecasts(actual, predicted):
#     '''
#     该函数实现根据预期值评估一个或多个周预测损失
#     思路：统计所有单日预测的 RMSE
#     '''
#     scores = list()
#     for i in range(actual.shape[1]):
#         mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
#         rmse = math.sqrt(mse)
#         scores.append(rmse)
#
#     s = 0  # 计算总的 RMSE
#     for row in range(actual.shape[0]):
#         for col in range(actual.shape[1]):
#             s += (actual[row, col] - predicted[row, col]) ** 2
#     score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
#     print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))
#     return score, scores

# def summarize_scores(name, score, scores):
#     s_scores = ', '.join(['%.1f' % s for s in scores])
#     print('%s: [%.3f] %s\n' % (name, score, s_scores))
#%%
training_set = data[:100000]
test_set = data[-6000:]
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)
print(len(training_set_scaled))
#%%
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)

        return array(X), array(y)
#%%
n_steps_in, n_steps_out = 30, 5
#%%
X_train, y_train = split_sequences(training_set_scaled, n_steps_in,n_steps_out)
X_test, y_test = split_sequences(testing_set_scaled, n_steps_in,n_steps_out)

#%%
n_features = X_train.shape[2]
#%%
from tensorflow.keras import regularizers
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=6, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(Conv1D(filters=64, kernel_size=6, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(200, activation='relu')))
model.add(TimeDistributed(Dense(n_features)))
# model.compile(loss='mse', optimizer='adam')
#%%
model.summary()  # 输出模型结构
#%%
# model = model.cuda()

#%%
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error',
              metrics=['accuracy'])  # 损失函数用均方误差
#%%
n_epochs = 100
# train model
history = model.fit(X_train,y_train,      #revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),  #revise
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()
#%%
model.save('new_factor_CNN_lstm_model_60_5.h5')
#%%
predicted_contract_price = model.predict(X_test)        # 测试集输入模型进行预测
#%%
def inverse_transform_col(scaler, y, n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= sc.min_[n_col]
    y /= sc.scale_[n_col]
    return y

#%%
# 第3列归一化的值
predict = predicted_contract_price[:, 0]

# 第3列反归一化
predict_col_0 = inverse_transform_col(sc, predict, n_col=0)

#%%
# 真实值返归一化
true = y_test[:,0]
true_col_0 = inverse_transform_col(sc, true, n_col=0)
#%%
truth = true_col_0[:, 0]
forecast = predict_col_0[:,0]
#%%
from sklearn.metrics import explained_variance_score

accuracy = explained_variance_score(truth, forecast)
print('准确率：%.5f' % accuracy)
#%%
from sklearn import metrics
'''
MSE  ：均方误差    ----->  预测值减真实值求平方后求均值
RMSE ：均方根误差  ----->  对均方误差开方
MAE  ：平均绝对误差----->  预测值减真实值求绝对值后求均值
R2   ：决定系数，可以简单理解为反映模型拟合优度的重要的统计量
'''
MSE = metrics.mean_squared_error(truth,forecast)
RMSE = metrics.mean_squared_error(truth,forecast)**0.5
MAE = metrics.mean_absolute_error(truth,forecast)
R2 = metrics.r2_score(truth,forecast)
# ACC = metrics.accuracy_score(true_col_0,predict_col_0)

print('均方误差: %.5f' % MSE)
print('均方根误差: %.5f' % RMSE)
print('平均绝对误差: %.5f' % MAE)
print('R2: %.5f' % R2)
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
#%%
true_result = pd.DataFrame(truth)
predict_result = pd.DataFrame(forecast)

test_set.astype(int)
true_result.astype(int)

prepare_data = test_set[34:]
#%%
col = ['predict']
predict_result.columns = col
col1 = ['close']
true_result.columns = col1
#%%
a = prepare_data.reset_index()
result = a.join(predict_result)
#%%
a = len(result[(result.log_return>0)&(result.predict>0)|((result.log_return<0)&(result.predict<0))])/len(result)