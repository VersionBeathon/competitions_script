#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time    :9:13
# @Author  :Kira
# @Name    :diabetes_predict_with_data_dealed
# @Software:PyCharm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor

train_path = 'H:\\阿里天池数据\\d_train_20180102_dealed.xlsx'
test_path = 'H:\\阿里天池数据\\d_test_A_20180102_dealed.xlsx'

sheet_name = ['f1f37', 'f1f20', 'f16f20']

train_data1 = pd.read_excel(train_path, sheetname=sheet_name[0])
train_data2 = pd.read_excel(train_path, sheetname=sheet_name[1])
train_data3 = pd.read_excel(train_path, sheetname=sheet_name[2])

test_data1 = pd.read_excel(test_path, sheetname=sheet_name[0])
test_data2 = pd.read_excel(test_path, sheetname=sheet_name[1])
test_data3 = pd.read_excel(test_path, sheetname=sheet_name[2])

base_header = ['sex', 'age']
header1 = base_header.copy()
[header1.append('f'+str(i)) for i in range(1, 38)]
header2 = base_header.copy()
[header2.append('f'+str(i)) for i in range(21, 38)]
header3 = base_header.copy()
[header3.append('f'+str(i)) for i in range(1, 38) if i <=15 or i >= 21]

trx_1 = train_data1[header1].copy()
try_1 = train_data1['y']
trx_2 = train_data2[header2].copy()
try_2 = train_data2['y']
trx_3 = train_data3[header3].copy()
try_3 = train_data3['y']

tsx_1 = test_data1[header1].copy()
tsx_2 = test_data2[header2].copy()
tsx_3 = test_data3[header3].copy()

replace_dict = {u'男': 1,
                u'女': 2,
                u'??': 3}
trx_1['sex'].replace(replace_dict, inplace=True)
trx_2['sex'].replace(replace_dict, inplace=True)
trx_3['sex'].replace(replace_dict, inplace=True)

tsx_1['sex'].replace(replace_dict, inplace=True)
tsx_2['sex'].replace(replace_dict, inplace=True)
tsx_3['sex'].replace(replace_dict, inplace=True)

for i in trx_1.columns:
    trx_1[i].fillna(trx_1[i].median(), inplace=True)
    tsx_1[i].fillna(tsx_1[i].median(), inplace=True)
for i in trx_2.columns:
    trx_2[i].fillna(trx_2[i].median(), inplace=True)
    tsx_2[i].fillna(tsx_2[i].median(), inplace=True)
for i in trx_3.columns:
    trx_3[i].fillna(trx_3[i].median(), inplace=True)
    tsx_3[i].fillna(tsx_3[i].median(), inplace=True)

rf = RandomForestRegressor(n_estimators=40, max_depth=5)
scores1 = []
for i in range(trx_1.shape[1]):
    score = cross_val_score(rf, trx_1.iloc[:, i:i+1], try_1, scoring='r2', cv=ShuffleSplit(len(trx_1), 1, .8))
    scores1.append((round(np.mean(score), 3), trx_1.columns[i]))
print(sorted(scores1, reverse=True))
feature_list1 = list()
for i in sorted(scores1, reverse=True):
    feature_list1.append(i[1])
trx_16 = trx_1[feature_list1[:6]].copy()
tsx_16 = tsx_1[feature_list1[:6]].copy()
svr1 = svm.SVR(C=0.8)
svr1.fit(trx_16, try_1)
tsy_1 = svr1.predict(tsx_16)


rf = RandomForestRegressor(n_estimators=40, max_depth=5)
scores2 = []
for i in range(trx_2.shape[1]):
    score = cross_val_score(rf, trx_2.iloc[:, i:i+1], try_2, scoring='r2', cv=ShuffleSplit(len(trx_2), 1, .8))
    scores2.append((round(np.mean(score), 3), trx_2.columns[i]))
print(sorted(scores2, reverse=True))
feature_list2 = list()
for i in sorted(scores2, reverse=True):
    feature_list2.append(i[1])
trx_26 = trx_2[feature_list2[:6]].copy()
tsx_26 = tsx_2[feature_list2[:6]].copy()
svr2 = svm.SVR(C=0.6)
svr2.fit(trx_26, try_2)
tsy_2 = svr2.predict(tsx_26)


rf = RandomForestRegressor(n_estimators=40, max_depth=5)
scores3 = []
for i in range(trx_3.shape[1]):
    score = cross_val_score(rf, trx_3.iloc[:, i:i+1], try_3, scoring='r2', cv=ShuffleSplit(len(trx_3), 1, .8))
    scores3.append((round(np.mean(score), 3), trx_3.columns[i]))
print(sorted(scores3, reverse=True))
feature_list3 = list()
for i in sorted(scores3, reverse=True):
    feature_list3.append(i[1])
trx_36 = trx_3[feature_list3[:6]].copy()
tsx_36 = tsx_3[feature_list3[:6]].copy()
svr3 = svm.SVR(C=0.6)
svr3.fit(trx_36, try_3)
tsy_3 = svr3.predict(tsx_36)

df1 = pd.DataFrame([test_data1['id'], tsy_1])
df2 = pd.DataFrame([test_data2['id'], tsy_2])
df3 = pd.DataFrame([test_data3['id'], tsy_3])

df1.T.to_csv('H:\\阿里天池数据\\part1_pred.csv')
df2.T.to_csv('H:\\阿里天池数据\\part2_pred.csv')
df3.T.to_csv('H:\\阿里天池数据\\part3_pred.csv')