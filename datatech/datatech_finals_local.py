#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time    :19:24
# @Author  :Kira
# @Name    :datatech_finals_local
# @Software:PyCharm
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score

train_path = 'C:\\Users\\Administrator\\workspace\\DataTech\\train_data.csv'
test_path = 'C:\\Users\\Administrator\\workspace\\DataTech\\test_data.csv'

part_train = pd.read_csv(train_path, header=None, names=['user_id', 'age', 'occupation_id', 'city_id', 'online_time',
                                                         'real_name_flag', 'user_credit_id', 'call_mark', 'comm_flag', 'tele_fac',
                                                         'tele_num', 'county_id', 'county_num', 'stop_days', 'stop_counts',
                                                         'call_counts', 'vpmn_call_counts', 'toll_counts', 'wj_call_counts', 'out_call_counts',
                                                         'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts', 'call_duration_m',
                                                         'bill_duration_m', 'vpmn_duration_m', 'wj_call_duration_m', 'out_call_duration_m', 'callfw_duration_m',
                                                         'bd_call_duration_m', 'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m', 'gprs_volume',
                                                         'arpu', 'sp_fee', 'num_of_comm', 'risk_flag',])


part_test = pd.read_csv(test_path, header=None, names=['user_id', 'age', 'occupation_id', 'city_id', 'online_time',
                                                       'real_name_flag', 'user_credit_id', 'call_mark', 'comm_flag', 'tele_fac',
                                                       'tele_num', 'county_id', 'county_num', 'stop_days', 'stop_counts',
                                                       'call_counts', 'vpmn_call_counts', 'toll_counts', 'wj_call_counts', 'out_call_counts',
                                                       'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts', 'call_duration_m',
                                                       'bill_duration_m', 'vpmn_duration_m', 'wj_call_duration_m', 'out_call_duration_m', 'callfw_duration_m',
                                                       'bd_call_duration_m', 'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m', 'gprs_volume',
                                                       'arpu', 'sp_fee', 'num_of_comm',
                                                       ])



part_train.replace('\\N', np.nan, inplace=True)
part_test.replace('\\N', np.nan, inplace=True)
X_part_train = part_train.loc[:,['age', 'occupation_id', 'city_id', 'online_time',
                                 'real_name_flag', 'user_credit_id', 'call_mark', 'comm_flag',
                                 'tele_num', 'county_num', 'stop_days', 'stop_counts',
                                 'call_counts', 'vpmn_call_counts', 'toll_counts', 'wj_call_counts', 'out_call_counts',
                                 'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts', 'call_duration_m',
                                 'bill_duration_m', 'vpmn_duration_m', 'wj_call_duration_m', 'out_call_duration_m', 'callfw_duration_m',
                                 'bd_call_duration_m', 'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m', 'gprs_volume',
                                 'arpu', 'sp_fee', 'num_of_comm',]]
y_part_train = part_train['risk_flag']

X_part_test = part_test.loc[:,['age', 'occupation_id', 'city_id', 'online_time',
                                 'real_name_flag', 'user_credit_id', 'call_mark', 'comm_flag',
                                 'tele_num', 'county_num', 'stop_days', 'stop_counts',
                                 'call_counts', 'vpmn_call_counts', 'toll_counts', 'wj_call_counts', 'out_call_counts',
                                 'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts', 'call_duration_m',
                                 'bill_duration_m', 'vpmn_duration_m', 'wj_call_duration_m', 'out_call_duration_m', 'callfw_duration_m',
                                 'bd_call_duration_m', 'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m', 'gprs_volume',
                                 'arpu', 'sp_fee', 'num_of_comm']]


# 标量处理
# tag_counts = ['call_counts', 'vpmn_call_counts', 'toll_counts','wj_call_counts', 'out_call_counts',
#              'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts']
# tag_duration = ['call_duration_m', 'bill_duration_m', 'vpmn_duration_m', 'wj_call_duration_m',
#                 'out_call_duration_m', 'callfw_duration_m', 'bd_call_duration_m', 'roam_duration_m',
#                 'toll_duration_m', 'qqw_call_duration_m', ]
# for i in tag_counts:
#     X_part_train.loc[X_part_train[i] > 9, i] = 9
# for j in tag_duration:
#     X_part_train.loc[X_part_train[j] > 9, j] = 9
# X_part_train.loc[X_part_train['gprs_volume'] > 10, 'gprs_volume'] = 10
# X_part_train.loc[X_part_train['num_of_comm'] > 5, 'num_of_comm'] = 5

# one_hot编码
y_final = np.empty([400000, 2])
for i in range(400000):
    if y_part_train[i] == 0:
        y_final[i, 0] = 1
        y_final[i, 1] = 0
    else:
        y_final[i, 0] = 0
        y_final[i, 1] = 1

X_part_train['age'].fillna(X_part_train['age'].median(), inplace=True)
X_part_train['sp_fee'].fillna(X_part_train['sp_fee'].median(), inplace=True)
X_part_train['arpu'].fillna(X_part_train['arpu'].mean(), inplace=True)
X_part_train['num_of_comm'].fillna(X_part_train['num_of_comm'].median(), inplace=True)
X_part_train['tele_num'].fillna(-1, inplace=True)

X_train = X_part_train_scaled = preprocessing.scale(X_part_train)
y_train = y_final

X_part_test['age'].fillna(X_part_test['age'].median(), inplace=True)
X_part_test['sp_fee'].fillna(X_part_test['sp_fee'].median(), inplace=True)
X_part_test['arpu'].fillna(X_part_test['arpu'].mean(), inplace=True)
X_part_test['num_of_comm'].fillna(X_part_test['num_of_comm'].median(), inplace=True)
X_part_test['tele_num'].fillna(-1, inplace=True)
X_test = preprocessing.scale(X_part_test)
# X_train, X_test, y_train, y_test = train_test_split(X_part_train_scaled, y_final, test_size=0.2)
# 构建神经网络
learning_rate = 0.01
train_epochs = 40000
n_sample = X_train.shape[0]
n_input = X_train.shape[1]
n_hidden_1 = 35
n_hidden_2 = 50
n_hidden_3 = 10
n_class = y_train.shape[1]
x = tf.placeholder(tf.float32, [None, n_input], name='x')
y = tf.placeholder(tf.float32, [None, n_class], name='y')
keep_prob = tf.placeholder(tf.float32)


def multiplayer_perceptron(x, weight, bias):
    layer1 = tf.nn.tanh(tf.add(tf.matmul(x, weight['h1']), bias['h1']))
    layer1_drop = tf.nn.dropout(layer1, keep_prob)
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1_drop, weight['h2']), bias['h2']))
    layer2_drop = tf.nn.dropout(layer2, keep_prob)
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2_drop, weight['h3']), bias['h3']))
    layer3_drop = tf.nn.dropout(layer3, keep_prob)
    out_layer =tf.nn.softmax(tf.add(tf.matmul(layer3_drop, weight['out']), bias['out']))
    return out_layer


weight = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_class])),
}


bias = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_class]))
}

pred = multiplayer_perceptron(x, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
# 运行搭建好的神经网络
with tf.Session() as sess:
    sess.run(init)
    correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_size = n_sample/train_epochs
    for epoch in range(train_epochs):
        batch_x, batch_y = X_train, y_train
        for batch in range(int(batch_size)):
            sess.run(optimizer, feed_dict={x: batch_x[batch*4000:(batch+1)*4000],
                                           y: batch_y[batch*4000:(batch+1)*4000],
                                           keep_prob: 0.8})
        # temp = sess.run(cost, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        # acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        # 每次训练后输入在测试集上的准确率
        # print(epoch, 'Accuracy:', acc)
        if epoch % 100 == 0:
            print('It have finished {0}'.format(epoch))
    print('Have finished training')
    # 存储模型
    save_path = saver.save(sess, "C:\\Users\\Administrator\\workspace\\DataTech\\model\\datatech_model.ckpt")
    # 模型最后在测试集上的准确率
    # print('final_arruracy:', sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0}))
    predictions = tf.arg_max(pred, 1)
    # 获取预测结果
    result = sess.run(predictions, feed_dict={x: X_test, keep_prob: 1.0})
    # f1值

result_final = pd.DataFrame(columns=['user_id', 'risk_flag']).astype('int')
for i in range(100000):
    result_final.loc[i] =[int(part_test['user_id'][i]), int(result[i])]

result_final.to_csv('C:\\Users\\Administrator\\workspace\\DataTech\\test_result.csv')
