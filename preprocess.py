#coding:utf-8
import pandas as pd
import time
import os
import numpy as np


def mean_list(list):
    mean = float(format(sum(list) / len(list)))
    return mean


def std_list(list):
    list_np = np.array(list)
    std = np.std(list_np, ddof=1)
    return std


def range_list(list):
    range = max(list) - min(list)
    return range


def median(x):
    x = sorted(x)
    length = len(x)
    mid, rem = divmod(length, 2)
    if rem:
        return x[:mid], x[mid + 1:], x[mid]
    else:
        return x[:mid], x[mid:], (x[mid - 1] + x[mid]) / 2


column_list = [
    'D101硫酸泵电流', 'T101到T102洗涤液流量',
    'E301干燥窑电流', 'T101进口稀磷酸流量',
    'E301干燥窑转速', 'T102Һλ',
    'E301干燥窑进口烟气温度', 'T102进口浓磷酸流量',
    'F101热风炉炉膛温度', 'Z220造粒机电流',
    'F101热风炉鼓风压力', 'Z221进口液氨压力',
    'F101鼓风机电流', 'Z221进口液氨流量',
    'G132洗涤塔液位', 'Z230进口洗涤液压力',
    'L101斗提机电流', 'Z230进口洗涤液浓度',
    'L102返料皮带电流', 'Z230进口液氨压力',
    'L103斗提机电流', 'Z230进口液氨流量',
    'S101A振网筛电流', 'Z230进口硫酸压力',
    'S101B振网筛电流', '内染剂泵流量',
    'S102A破碎机电流1', '成品重量',
    'S102A破碎机电流2', '浓硫酸流量',
    'S102B破碎机电流1', '液氨压力',
    'S102B破碎机电流2', '液氨流量',
    'T101Һλ', '返料重量']
b = '/home/data/2.产品检验报告/产品检验报告2018-4-1.csv'
dfb_new = pd.read_csv(b)

for column_name in column_list:
    dfa = pd.read_csv(os.path.join('/home/data/1.生产参数记录表/生产参数记录表-2018年4月', column_name + '-Raw.csv'), header='infer')

    dfa.columns = ['id', 'time', 'parameter']
    dfb_new[column_name + '_mean'] = None
    # dfb_new[column_name + '_std'] = None
    # dfb_new[column_name + '_max'] = None
    # dfb_new[column_name + '_min'] = None
    # dfb_new[column_name + '_range'] = None
    # dfb_new[column_name + '_Q1'] = None
    # dfb_new[column_name + '_Q2'] = None
    # dfb_new[column_name + '_Q3'] = None
    for index in range(dfb_new.shape[0]):
        product_batch = dfb_new.loc[index, 'product_batch']
        datestr = '2018.' + product_batch.split(' ')[0]
        begin = product_batch.split(' ')[1] + ':00'
        begin_time = time.mktime(time.strptime(datestr + ' ' + begin, '%Y.%m.%d %H:%M:%S'))
        end = product_batch.split(' ')[3] + ':00'
        end_time = time.mktime(time.strptime(datestr + ' ' + end, '%Y.%m.%d %H:%M:%S'))
        if end == '00:00:00':
            end_time = end_time + 86400
        dfa_pra = []

        for i in range(dfa.shape[0]):
            try:
                obeserve_time = time.mktime(time.strptime(dfa.loc[i, 'time'][1:20], '%Y-%m-%d %H:%M:%S'))
            except TypeError:
                continue
            if obeserve_time >= begin_time and obeserve_time < end_time:
                dfa_pra.append(dfa.loc[i, 'parameter'])
            elif obeserve_time >= end_time:
                dfa.drop([row for row in range(i)], inplace=True)
                dfa = dfa.reset_index(drop=True)
                break
        if len(dfa_pra) > 0:
            # Q1, Q3, Q2 = median(dfa_pra)
            dfb_new.loc[index, column_name + '_mean'] = mean_list(dfa_pra)
            # dfb_new.loc[index, column_name + '_std'] = std_list(dfa_pra)
            # dfb_new.loc[index, column_name + '_max'] = max(dfa_pra)
            # dfb_new.loc[index, column_name + '_min'] = min(dfa_pra)
            # dfb_new.loc[index, column_name + '_range'] = range_list(dfa_pra)
            # dfb_new.loc[index, column_name + '_Q1'] = median(Q1)[2]
            # dfb_new.loc[index, column_name + '_Q2'] = Q2
            # dfb_new.loc[index, column_name + '_Q3'] = median(Q3)[2]
    print(column_name + ' done')
dfb_new.to_csv('train_mean.csv', index=False)

c = '/home/data/2.产品检验报告/产品检验报告2018-5-1-sample.csv'
dfc_new = pd.read_csv(c)

for column_name in column_list:
    dfa = pd.read_csv(os.path.join('/home/data/1.生产参数记录表/生产参数记录表-2018年5月', column_name + '-Raw.csv'), header='infer')

    dfa.columns = ['id', 'time', 'parameter']
    dfc_new[column_name + '_mean'] = None
    # dfc_new[column_name + '_std'] = None
    # dfc_new[column_name + '_max'] = None
    # dfc_new[column_name + '_min'] = None
    # dfc_new[column_name + '_range'] = None
    # dfc_new[column_name + '_Q1'] = None
    # dfc_new[column_name + '_Q2'] = None
    # dfc_new[column_name + '_Q3'] = None
    for index in range(dfc_new.shape[0]):
        product_batch = dfc_new.loc[index, 'product_batch']
        datestr = '2018.' + product_batch.split(' ')[0]
        begin = product_batch.split(' ')[1] + ':00'
        begin_time = time.mktime(time.strptime(datestr + ' ' + begin, '%Y.%m.%d %H:%M:%S'))
        end = product_batch.split(' ')[3] + ':00'
        end_time = time.mktime(time.strptime(datestr + ' ' + end, '%Y.%m.%d %H:%M:%S'))
        if end == '00:00:00':
            end_time = end_time + 86400

        dfa_pra = []
        for i in range(dfa.shape[0]):
            obeserve_time = time.mktime(time.strptime(dfa.loc[i, 'time'][1:20], '%Y-%m-%d %H:%M:%S'))
            if obeserve_time >= begin_time and obeserve_time < end_time:
                dfa_pra.append(dfa.loc[i, 'parameter'])
            elif obeserve_time >= end_time:
                dfa.drop([row for row in range(i)], inplace=True)
                dfa = dfa.reset_index(drop=True)
                break
        if len(dfa_pra) > 0:
            Q1, Q3, Q2 = median(dfa_pra)
            dfc_new.loc[index, column_name + '_mean'] = mean_list(dfa_pra)
            # dfc_new.loc[index, column_name + '_std'] = std_list(dfa_pra)
            # dfc_new.loc[index, column_name + '_max'] = max(dfa_pra)
            # dfc_new.loc[index, column_name + '_min'] = min(dfa_pra)
            # dfc_new.loc[index, column_name + '_range'] = range_list(dfa_pra)
            # dfc_new.loc[index, column_name + '_Q1'] = median(Q1)[2]
            # dfc_new.loc[index, column_name + '_Q2'] = Q2
            # dfc_new.loc[index, column_name + '_Q3'] = median(Q3)[2]

    print(column_name + ' done')

dfc_new.to_csv('test_mean.csv', index=False)
