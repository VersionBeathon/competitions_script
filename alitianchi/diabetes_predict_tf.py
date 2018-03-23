#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time    :10:03
# @Author  :Kira
# @Name    :diabetes_predict_tf
# @Software:PyCharm
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


input_path = 'H:\\阿里天池数据\\d_train.csv'
train_data = pd.read_csv(input_path, encoding='gbk')
# 删除date字段
train_data.drop(['date'], axis=1, inplace=True)
replace_dict = {u'男': 1,
                u'女': 2,
                u'??': 3}
# 将sex字段进行替换
train_data['sex'].replace(replace_dict, inplace=True)

part_x = train_data[train_data.columns[1:40]].copy()
part_y = train_data['y'].copy()
# 缺失值补全，暂时用中位数
for i in part_x.columns:
    part_x[i].fillna(part_x[i].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(part_x, part_y, test_size=0.2)

x = tf.placeholder(tf.float32, [None, 39])
W = tf.Variable(tf.zeros([39, 1]))
b = tf.Variable(tf.zeros([1]))
y_ =  tf.add(tf.matmul(x, W), b)
y = tf.placeholder(tf.float32, [None, 1])

lost = tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_step = optimizer.minimize(lost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

steps = 1000
for i in range(steps):
    feed = {x: X_train, y: np.asarray(y_train).reshape((y_train.shape[0], 1))}
    sess.run(train_step, feed_dict=feed)
    if i % 100 == 0:
        print('lost:{0}'.format(sess.run(lost, feed_dict=feed)))


X = tf.placeholder(tf.float32, [None, 39])
w = tf.Variable(tf.zeros([39, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
Y = tf.placeholder(tf.float32, [None, 1])

# 成本函数 sum(sqr(y_-y))/n
cost = tf.reduce_mean(tf.square(Y-y))

# 用梯度下降训练
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# x_train = [[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]]
# y_train = [[7], [8], [10], [14], [8], [13], [20], [16], [28], [26]]

for i in range(10000):
    feed = {X: X_train, Y: np.asarray(y_train).reshape((y_train.shape[0], 1))}
    sess.run(train_step, feed_dict=feed)
    if i % 100 == 0:
        print('lost:{0}'.format(sess.run(cost, feed_dict=feed)))
print("w0:%f" % sess.run(w[0]))
print("w1:%f" % sess.run(w[1]))
print("b:%f" % sess.run(b))