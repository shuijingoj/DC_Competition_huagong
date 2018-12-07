import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut

###################两个模型结果再训练##################

train_data_df_lgb = pd.read_csv('train_data_lgb.csv', header='infer', encoding='utf-8')
test_data_df_lgb = pd.read_csv('test_data_lgb.csv', header='infer', encoding='utf-8')

train_data_df_svr = pd.read_csv('train_data_svr.csv', header='infer', encoding='utf-8')
test_data_df_svr = pd.read_csv('test_data_svr.csv', header='infer', encoding='utf-8')

train_data_df = pd.concat([train_data_df_lgb.iloc[:, 1:], train_data_df_svr.iloc[:, 1:]], axis=1)
test_data_df = pd.concat([test_data_df_lgb.iloc[:, 1:], test_data_df_svr.iloc[:, 1:]], axis=1)

X_train = train_data_df.values
X_test = test_data_df.values

train_label_file = '/home/data/2.产品检验报告/产品检验报告2018-4-1.csv'
test_label_file = '/home/data/2.产品检验报告/产品检验报告2018-5-1-sample.csv'

y_train_df = pd.read_csv(train_label_file, header='infer', encoding='gbk')
y_test_df = pd.read_csv(test_label_file, header='infer', encoding='gbk')
y_train = y_train_df.iloc[:, 2:7].values.T
y_test = y_test_df.iloc[:, 2:7].values.T

###MLP 留一法验证模型，全数据集训练
label_list = y_test_df.columns.values[2:].tolist()
score = 0
result = {}
for i in range(len(label_list)):
    k = 0
    rmse = []
    loo = LeaveOneOut()  # 留一法划分数据集
    for train_index, valid_index in loo.split(X_train):
        k += 1
        X_train_t, X_valid_t = X_train[train_index], X_train[valid_index]
        y_train_t, y_valid_t = y_train.T[train_index], y_train.T[valid_index]
        model = MLPRegressor(hidden_layer_sizes=(20), max_iter=500, activation='relu', solver='lbfgs', random_state=0)
        model.fit(X_train_t, y_train_t.T[i])
        y_pred_valid = model.predict(X_valid_t)
        rmse.append(mean_squared_error(y_valid_t.T[i], y_pred_valid) ** 0.5)
    score = np.mean(rmse)
    print('The RMSE on validation set of %s: %.5f' % (label_list[i], score))

'''使用模型最佳参数得到结果'''
num_iter = [500, 500, 500, 500, 100]
for i in range(len(label_list)):  # 为每个指标单独训练一个模型
    model_mlp = MLPRegressor(hidden_layer_sizes=(20), max_iter=num_iter[i], activation='relu', solver='lbfgs',
                             random_state=0)
    model_mlp.fit(X_train, y_train[i])
    result[label_list[i]] = model_mlp.predict(X_test)

result_df = pd.DataFrame.copy(y_test_df)
for col in result.keys():
    result_df[col] = result[col]

result_df = result_df.drop(['product_batch'], axis=1)
result_df['total_nutrient'] = result_df['phosphorus_content'] + result_df['nitrogen_content']
result_df.to_csv('result.csv', index=False, encoding='utf-8')
