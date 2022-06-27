#%%
import datetime

# import matplotlib.sphinxext.plot_directive
import keras
import tensorflow as tf
import pandas as pd

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from numpy import array
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


#%%
data = pd.read_csv('data_2min_bar_2020_2021_new.csv')
# a = pd.read_csv('data_2min_bar_2020_2021.csv')
#%%
# data['time'] = pd.to_datetime(data['time'])
data = data.set_index(['time'])
# a = a.set_index(['time'])
#%%
data = data.drop(['sellprice','buyprice','pricemean','tradingprice'],axis=1)
#%%
data = data[~data['close'].isin([0])]
#%%
training_set = data[(data.index>='2020-01-08 0:00:00')& (data.index<='2021-07-01 0:00:00')]
test_set = data[(data.index>='2021-05-01 0:00:00')& (data.index<='2021-11-10 23:59:59')]
#%%
test_data = data[(data.index>='2021-01-01 0:00:00')& (data.index<='2021-11-12 0:00:00')]
#%%
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)
# test_data_scaled = sc.transform(test_data)
# print(len(training_set_scaled))

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
n_steps_in, n_steps_out = 60, 5
#%%
X_train, y_train = split_sequences(training_set_scaled, n_steps_in,n_steps_out)
X_test, y_test = split_sequences(testing_set_scaled, n_steps_in,n_steps_out)
#%%
X_test,y_test = split_sequences(test_data_scaled, n_steps_in,n_steps_out)
#%%
# from tensorflow import keras
from tensorflow.keras.models import load_model
#%%
model = load_model('new_factor_CNN_lstm_model_60_5.h5')
#%%
# activate model
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
#%%
# train model
history = model.fit(X_train, y_train,
                    batch_size=128,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()
#%%
model.save('CNN_lstm_model_60_5step_2021.h5')
#%%
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Val_loss.png')
plt.show()
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
predict = predicted_contract_price[:, -1]

# 第3列反归一化
predict_col_0 = inverse_transform_col(sc, predict, n_col=-1)

forecast = predict_col_0[:,-1]
#%%
# 真实值返归一化
true = y_test[:,-1]
true_col_0 = inverse_transform_col(sc, true, n_col=-1)
truth = true_col_0[:,-1]
# #%%
# predicted_price = sc.inverse_transform(predicted_contract_price) # 对预测数据还原---从（0，1）反归一化到原始范围
# #%%
# real_price = sc.inverse_transform(y_test)   # 对真实数据还原---从（0，1）反归一化到原始范围
#%%
print('RMSE', np.sqrt(metrics.mean_squared_error(forecast, truth)))

#%%
'''
MSE  ：均方误差    ----->  预测值减真实值求平方后求均值
RMSE ：均方根误差  ----->  对均方误差开方
MAE  ：平均绝对误差----->  预测值减真实值求绝对值后求均值
R2   ：决定系数，可以简单理解为反映模型拟合优度的重要的统计量
# '''
# MSE = metrics.mean_squared_error(predict_col_0[27000:], true_col_0[27000:])
# RMSE = metrics.mean_squared_error(predict_col_0[27000:], true_col_0[27000:])**0.5
# MAE = metrics.mean_absolute_error(predict_col_0[27000:], true_col_0[27000:])
# R2 = metrics.r2_score(predict_col_0[27000:], true_col_0[27000:])
MSE = metrics.mean_squared_error(truth,forecast)
RMSE = metrics.mean_squared_error(truth,forecast)**0.5
MAE = metrics.mean_absolute_error(truth,forecast)
R2 = metrics.r2_score(truth,forecast)

print('均方误差: %.5f' % MSE)
print('均方根误差: %.5f' % RMSE)
print('平均绝对误差: %.5f' % MAE)
print('R2: %.5f' % R2)
#%%
plt.plot(true_col_0[27900:]-predict_col_0[27900:], color= 'black')
plt.title('difference')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.savefig('difference_2021.png')
plt.show()
#%%
true_result = pd.DataFrame(truth)
# true_result.to_csv('result.csv')
#%%
predict_result = pd.DataFrame(forecast)

#%%
test_set.astype(int)
true_result.astype(int)
#%%
prepare_data = test_data[64:]

#%%
col = ['predict']
predict_result.columns = col
col1 = ['close']
true_result.columns = col1
#%%
a = prepare_data.reset_index()
result = a.join(predict_result)
result['close'] = result['predict']
result = result.drop(['predict'], axis=1)
#%%
result.to_csv('predictvalue_CNN_LSTM_60_5step_12month.csv')
a.to_csv('truevalue_CNN_LSTM_60_5step_12month.csv')

#%%
