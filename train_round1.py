import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR

import warnings

warnings.filterwarnings("ignore")

# Train 1: 准备训练数据，请先运行预处理模块产生所需数据
train_data_df = pd.read_csv('train_mean.csv', header='infer', encoding='utf-8')
test_data_df = pd.read_csv('test_mean.csv', header='infer', encoding='utf-8')
'''添加删减新特征'''

train_data_df['返料比'] = train_data_df['返料重量_mean'] / train_data_df['成品重量_mean']
test_data_df['返料比'] = test_data_df['返料重量_mean'] / test_data_df['成品重量_mean']
train_data_df['返料差'] = train_data_df['返料重量_mean'] - train_data_df['成品重量_mean']
test_data_df['返料差'] = test_data_df['返料重量_mean'] - test_data_df['成品重量_mean']


X_train = train_data_df.iloc[:, 7:].values
X_test = test_data_df.iloc[:, 7:].values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 以均值填补缺失值
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

train_label_file = '/home/data/2.产品检验报告/产品检验报告2018-4-1.csv'
test_label_file = '/home/data/2.产品检验报告/产品检验报告2018-5-1-sample.csv'
y_train_df = pd.read_csv(train_label_file, header='infer', encoding='gbk')
y_test_df = pd.read_csv(test_label_file, header='infer', encoding='gbk')
y_train = y_train_df.iloc[:, 2:7].values.T
y_test = y_test_df.iloc[:, 2:7].values.T

label_list = y_test_df.columns.values[2:].tolist()
feature_list = train_data_df.columns[7:]

#####################################################################3
# Train 2-1*: lightGBM 留一法验证模型，全数据集训练
'''获取每个模型最佳参数'''

for i in range(len(label_list)):
    k = 0
    rmse = []
    loo = LeaveOneOut()  # 留一法划分数据集
    for train_index, valid_index in loo.split(X_train):
        k += 1
        X_train_t, X_valid_t = X_train[train_index], X_train[valid_index]
        y_train_t, y_valid_t = y_train.T[train_index], y_train.T[valid_index]
        model = lgb.LGBMRegressor(objective='regression', max_depth=4, num_leaves=32, learning_rate=0.05,
                                  n_estimators=100, verbose=-1)
        model.fit(X_train_t, y_train_t.T[i])
        y_pred_valid = model.predict(X_valid_t)
        rmse.append(mean_squared_error(y_valid_t.T[i], y_pred_valid) ** 0.5)  # 计算在验证集上的RMSE

    score = np.mean(rmse)
    print('The RMSE of LGBR on validation set of %s: %.5f' % (label_list[i], score))

######################################################################################
# Train 2-2*: SVR 留一法验证模型，全数据集训练
for i in range(len(label_list)):
    k = 0
    rmse = []
    loo = LeaveOneOut()  # 留一法划分数据集
    for train_index, valid_index in loo.split(X_train):
        k += 1
        X_train_t, X_valid_t = X_train[train_index], X_train[valid_index]
        y_train_t, y_valid_t = y_train.T[train_index], y_train.T[valid_index]

        model = SVR(gamma='scale')
        model.fit(X_train_t, y_train_t.T[i])
        y_pred_valid = model.predict(X_valid_t)
        rmse.append(mean_squared_error(y_valid_t.T[i], y_pred_valid) ** 0.5)  # 计算在验证集上的RMSE
    #         if k%10 == 0:
    #             print('training %s model %d times...'%(label_list[i], k))
    score = np.mean(rmse)
    print('The RMSE of SVR on validation set of %s: %.5f' % (label_list[i], score))


#################################################################################################################
'''使用模型最佳参数得到测试集和训练集的结果'''

model_lgb = [lgb.LGBMRegressor(objective='regression', max_depth=3, num_leaves=32, learning_rate=0.05, n_estimators=110,
                               verbose=-1, feature_fraction=0.7),
             lgb.LGBMRegressor(objective='regression', max_depth=4, num_leaves=32, learning_rate=0.02, n_estimators=180,
                               verbose=-1),
             lgb.LGBMRegressor(objective='regression', max_depth=2, num_leaves=32, learning_rate=0.05, n_estimators=40,
                               verbose=-1),
             lgb.LGBMRegressor(objective='regression', max_depth=1, num_leaves=32, learning_rate=0.05, n_estimators=110,
                               verbose=-1),
             lgb.LGBMRegressor(objective='regression', max_depth=1, num_leaves=32, learning_rate=0.03, n_estimators=110,
                               verbose=-1)
             ]

model_svr = SVR(gamma='scale')


########################lgb的测试集结果###########################
result_test = {}
for i in range(len(label_list)):
    model_lgb[i].fit(X_train, y_train[i])
    result_test[label_list[i]] = model_lgb[i].predict(X_test)
result_df = pd.DataFrame.copy(y_test_df)
for col in result_test.keys():
    result_df[col] = result_test[col]
result_df = result_df.drop(['product_batch'], axis=1)
result_df.to_csv('test_data_lgb.csv', index=False, encoding='utf-8')

#########################lgb的训练集结果###########################
result_train = {}
for i in range(len(label_list)):
    model_lgb[i].fit(X_train, y_train[i])
    result_train[label_list[i]] = model_lgb[i].predict(X_train)
result_df = pd.DataFrame.copy(y_train_df)
for col in result_train.keys():
    result_df[col] = result_train[col]
result_df = result_df.drop(['product_batch'], axis=1)
result_df.to_csv('train_data_lgb.csv', index=False, encoding='utf-8')

########################svr的测试集结果############################
result_test = {}
for i in range(len(label_list)):
    model_svr.fit(X_train, y_train[i])
    result_test[label_list[i]] = model_svr.predict(X_test)
result_df = pd.DataFrame.copy(y_test_df)
for col in result_test.keys():
    result_df[col] = result_test[col]
result_df = result_df.drop(['product_batch'], axis=1)
result_df.to_csv('test_data_svr.csv', index=False, encoding='utf-8')

########################svr的训练集结果############################
result_train = {}
for i in range(len(label_list)):
    model_svr.fit(X_train, y_train[i])
    result_train[label_list[i]] = model_svr.predict(X_train)
result_df = pd.DataFrame.copy(y_train_df)
for col in result_train.keys():
    result_df[col] = result_train[col]
result_df = result_df.drop(['product_batch'], axis=1)
result_df.to_csv('train_data_svr.csv', index=False, encoding='utf-8')

