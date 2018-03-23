#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time    :9:25
# @Author  :Kira
# @Name    :data_tech_preliminary
# @Software:PyCharm
import tensorflow as tf
import pandas as pd
import numpy as np
import pyexcel as pe
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score

np.set_printoptions(threshold=10000000, linewidth=1000)

file_com = 'H:\DataTech\DataTech模型赛Sample-选题三-ChinaMobile\DataTech_Credit_Train_Communication1.txt'
file_user = 'H:\DataTech\DataTech模型赛Sample-选题三-ChinaMobile\DataTech_Credit_Train_User1.txt'
file_base = 'H:\DataTech\DataTech模型赛Sample-公共数据集-ChinaMobile\DataTech_公共数据_基础信息1.txt'
file_tele_type = 'H:\DataTech\手机型号.xlsx'
part_com = pd.read_table(file_com, header=None, names=['id', 'days', 'counts', 'date'], sep=',', index_col=None)
part_user = pd.read_table(file_user, header=None, names=['id', 'flag'], sep=',', index_col=None)
part_base = pd.read_table(file_base, header=None, names=['id', 'age', 'occupation_id', 'city_id', 'county_id',
                                                         'online_time', 'real_name_flag', 'user_credit_id', 'call_mark',
                                                         'comm_flag', 'call_counts', 'vpmn_call_counts', 'toll_counts',
                                                         'wj_call_counts', 'out_call_counts', 'callfw_counts', 'qqw_call_counts',
                                                         'bd_call_counts', 'roam_counts', 'call_duration_m', 'bill_duration_m',
                                                         'vpmn_call_duration_m', 'wj_call_duration_m', 'out_call_duration_m',
                                                         'callfw_duration_m', 'bd_call_duration_m',
                                                         'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m',
                                                         'gprs_volume', 'arpu', 'sp_fee', 'tele_type', 'tele_fac', 'smart_system',
                                                         'fist_use_date', 'num_of_comm', 'date'], sep=',', index_col=None)
tele_dict = dict()
data_tele = pe.get_array(file_name=file_tele_type)
for i in data_tele:
    tele_dict[i[0]] = i[1]

county_dict = dict()
county_list = list(set(part_base['county_id']))
county_list.sort()
num = 0
for i in county_list:
    county_dict[i] = num
    num += 1


part_base.replace('\\N', np.nan, inplace=True)
part_base_select = part_base.loc[:,['id', 'age', 'occupation_id', 'city_id',
                                    'online_time', 'real_name_flag', 'user_credit_id',
                                    'call_mark', 'comm_flag', 'call_counts', 'vpmn_call_counts', 'toll_counts',
                                    'wj_call_counts', 'out_call_counts', 'callfw_counts', 'qqw_call_counts',
                                    'bd_call_counts', 'roam_counts', 'call_duration_m', 'bill_duration_m',
                                    'vpmn_call_duration_m', 'wj_call_duration_m', 'out_call_duration_m',
                                    'callfw_duration_m', 'bd_call_duration_m',
                                    'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m',
                                    'gprs_volume', 'arpu', 'sp_fee', 'num_of_comm','date']]
part_base_select.astype("float", inplace=True)
df_base_one = pd.DataFrame(columns=['id', 'age', 'occupation_id', 'city_id',
                                    'online_time', 'real_name_flag', 'user_credit_id',
                                    'call_mark', 'comm_flag'])
df_base_two = pd.DataFrame(columns=['id', 'call_counts', 'vpmn_call_counts', 'toll_counts',
                                    'wj_call_counts', 'out_call_counts', 'callfw_counts', 'qqw_call_counts',
                                    'bd_call_counts', 'roam_counts', 'call_duration_m', 'bill_duration_m',
                                    'vpmn_call_duration_m', 'wj_call_duration_m', 'out_call_duration_m',
                                    'callfw_duration_m', 'bd_call_duration_m',
                                    'roam_duration_m', 'toll_duration_m', 'qqw_call_duration_m',
                                    'gprs_volume', 'arpu', 'sp_fee', 'num_of_comm'])

df_tele = pd.DataFrame(columns=['id', 'tele_fac'])
df_county = pd.DataFrame(columns=['id', 'county_id'])
#处理归属县市
signal_county = part_base['id'].min()
county_index_counts = 0
for a, b in part_base.sort_values(by=['id', 'date']).iterrows():
    if signal_county != b['id']:
        county_index_counts += 1
        signal_county = b['id']
    else:
        try:
            df_county.loc[county_index_counts] = [b['id'], county_dict[b['county_id']]]
        except:
            df_county.loc[county_index_counts] = [b['id'], -1]
#处理电话类型
signal_tele = part_base['id'].min()
tele_index_counts = 0
for q, w in part_base.sort_values(by=['id', 'date']).iterrows():
    if signal_tele != w['id']:
        tele_index_counts += 1
        signal_tele = w['id']
    else:
        try:
            df_tele.loc[tele_index_counts] = [w['id'], tele_dict[w['tele_fac']]]
        except:
            df_tele.loc[tele_index_counts] = [w['id'], -1]

# 处理base_one
one_index = 0
signal_one = part_base_select['id'].min()
for i, j in part_base_select.sort_values(by=['id', 'date']).iterrows():
    if signal_one != j['id']:
        one_index += 1
        signal_one = j['id']
    else:
        df_base_one.loc[one_index] = [j['id'], j['age'], j['occupation_id'], j['city_id'], j['online_time'],
                                  j['real_name_flag'], j['user_credit_id'], j['call_mark'], j['comm_flag']]

# 处理base_two
two_index = 0
signal_two = part_base_select['id'].min()
call_counts = 0
vpmn_call_counts = 0
toll_counts = 0
wj_call_counts = 0
out_call_counts = 0
callfw_counts = 0
qqw_call_counts = 0
bd_call_counts = 0
roam_counts = 0
call_duration_m = 0
bill_duration_m = 0
vpmn_call_duration_m = 0
wj_call_duration_m = 0
out_call_duration_m = 0
callfw_duration_m = 0
bd_call_duration_m = 0
roam_duration_m = 0
toll_duration_m = 0
qqw_call_duration_m = 0
gprs_volume = 0
arpu = 0
sp_fee = 0
num_of_comm = 0
for m, n in part_base_select.sort_values(by=['id', 'date']).astype('float').iterrows():
    if signal_two == n['id']:
        call_counts += n['call_counts']
        vpmn_call_counts += n['vpmn_call_counts']
        toll_counts += n['toll_counts']
        wj_call_counts += n['wj_call_counts']
        out_call_counts += n['out_call_counts']
        callfw_counts += n['callfw_counts']
        qqw_call_counts += n['qqw_call_counts']
        bd_call_counts += n['bd_call_counts']
        roam_counts += n['roam_counts']
        call_duration_m += n['call_duration_m']
        bill_duration_m += n['bill_duration_m']
        vpmn_call_duration_m += n['vpmn_call_duration_m']
        wj_call_duration_m += n['wj_call_duration_m']
        out_call_duration_m += n['out_call_duration_m']
        callfw_duration_m += n['callfw_duration_m']
        bd_call_duration_m += n['bd_call_duration_m']
        roam_duration_m += n['roam_duration_m']
        toll_duration_m += n['toll_duration_m']
        qqw_call_duration_m += n['qqw_call_duration_m']
        gprs_volume  += n['gprs_volume']
        arpu += n['arpu']
        sp_fee += n['sp_fee']
        num_of_comm = n['num_of_comm']
    else:
        df_base_two.loc[two_index] = [signal_two, call_counts, vpmn_call_counts, toll_counts, wj_call_counts,
                                  out_call_counts, callfw_counts, qqw_call_counts, bd_call_counts, roam_counts,
                                  call_duration_m, bill_duration_m, vpmn_call_duration_m, wj_call_duration_m,
                                  out_call_duration_m, callfw_duration_m, bd_call_duration_m, roam_duration_m,
                                  toll_duration_m, qqw_call_duration_m, gprs_volume, arpu, sp_fee, num_of_comm]
        two_index += 1
        signal_two = n['id']
        call_counts = 0
        vpmn_call_counts = 0
        toll_counts = 0
        wj_call_counts = 0
        out_call_counts = 0
        callfw_counts = 0
        qqw_call_counts = 0
        bd_call_counts = 0
        roam_counts = 0
        call_duration_m = 0
        bill_duration_m = 0
        vpmn_call_duration_m = 0
        wj_call_duration_m = 0
        out_call_duration_m = 0
        callfw_duration_m = 0
        bd_call_duration_m = 0
        roam_duration_m = 0
        toll_duration_m = 0
        qqw_call_duration_m = 0
        gprs_volume = 0
        arpu = 0
        sp_fee = 0
df_base_two.loc[two_index+1] = [signal_two, call_counts, vpmn_call_counts, toll_counts, wj_call_counts,
                          out_call_counts, callfw_counts, qqw_call_counts, bd_call_counts, roam_counts,
                          call_duration_m, bill_duration_m, vpmn_call_duration_m, wj_call_duration_m,
                          out_call_duration_m, callfw_duration_m, bd_call_duration_m, roam_duration_m,
                          toll_duration_m, qqw_call_duration_m, gprs_volume, arpu, sp_fee, num_of_comm]

# 处理part_com
df_com = pd.DataFrame(columns=['id', 'total_days', 'total_counts'])
total_days = 0
total_counts = 0
user_signal = part_com['id'].min()
user_index = 0
for k, r in part_com.sort_values(by='id').iterrows():
    if user_signal == r['id']:
        total_days += r['days']
        total_counts += r['counts']
    else:
        df_com.loc[user_index] = [user_signal, total_days, total_counts]
        user_index += 1
        user_signal = r['id']
        total_days = 0
        total_counts = 0
df_com.loc[user_index+1] = [user_signal, total_days, total_days]

# 合并数据
df_one = pd.merge(part_user, df_base_one, how='left', on=['id'])
df_two = pd.merge(df_one, df_base_two, how='left', on=['id'])
df_three = pd.merge(df_two, df_com, how='left', on=['id'])
df_four = pd.merge(df_three, df_tele, how='left', on=['id'])
df_five = pd.merge(df_four, df_county, how='left', on=['id'])
X = df_five.loc[:,['age', 'occupation_id', 'city_id', 'online_time', 'real_name_flag', 'user_credit_id','call_mark',
                   'comm_flag', 'call_counts', 'vpmn_call_counts', 'toll_counts','wj_call_counts', 'out_call_counts',
                   'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts', 'call_duration_m',
                   'bill_duration_m', 'vpmn_call_duration_m', 'wj_call_duration_m', 'out_call_duration_m',
                   'callfw_duration_m', 'bd_call_duration_m', 'roam_duration_m', 'toll_duration_m',
                   'qqw_call_duration_m', 'gprs_volume', 'arpu', 'sp_fee', 'num_of_comm',
                   'total_days', 'total_counts', 'tele_fac', 'county_id']]
y = df_five['flag']
# 缺失值处理
X['age'].fillna(X['age'].median(), inplace=True)
X['sp_fee'].fillna(X['sp_fee'].median(), inplace=True)
X['arpu'].fillna(X['arpu'].mean(), inplace=True)
X['num_of_comm'].fillna(X['num_of_comm'].median(), inplace=True)
# 标量处理
tag_counts = ['call_counts', 'vpmn_call_counts', 'toll_counts','wj_call_counts', 'out_call_counts',
             'callfw_counts', 'qqw_call_counts', 'bd_call_counts', 'roam_counts']
tag_duration = ['call_duration_m', 'bill_duration_m', 'vpmn_call_duration_m', 'wj_call_duration_m',
                'out_call_duration_m', 'callfw_duration_m', 'bd_call_duration_m', 'roam_duration_m',
                'toll_duration_m', 'qqw_call_duration_m', ]
for i in tag_counts:
    X.loc[X[i] > 9, i] = 9
for j in tag_duration:
    X.loc[X[j] > 9, j] = 9
X.loc[X['gprs_volume'] > 10, 'gprs_volume'] = 10
X.loc[X['num_of_comm'] > 5, 'num_of_comm'] = 5
# one_hot编码
y_final = np.empty([7000, 2])
for i in range(7000):
    if y[i] == 0:
        y_final[i, 0] = 1
        y_final[i, 1] = 0
    else:
        y_final[i, 0] = 0
        y_final[i, 1] = 1
# 归一化
X_pca = preprocessing.scale(X)
# 划分训练集跟测试集 训练集百分之80 测试集百分之20
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_final, test_size=0.2)

# 构建神经网络
learning_rate = 0.08
train_epochs = 10000
n_sample = X_train.shape[0]
n_input = X_train.shape[1]
n_hidden_1 = 35
n_hidden_2 = 50
n_hidden_3 = 10
n_class = y_train.shape[1]
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32)


def multiplayer_perceptron(x, weight, bias):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weight['h1']), bias['h1']))
    layer1_drop = tf.nn.dropout(layer1, keep_prob)
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1_drop, weight['h2']), bias['h2']))
    layer2_drop = tf.nn.dropout(layer2, keep_prob)
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2_drop, weight['h3']), bias['h3']))
    layer3_drop = tf.nn.dropout(layer3, keep_prob)
    out_layer =tf.nn.sigmoid(tf.add(tf.matmul(layer3_drop, weight['out']), bias['out']))
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

# 运行搭建好的神经网络
with tf.Session() as sess:
    sess.run(init)
    correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for epoch in range(train_epochs):
        batch_x, batch_y = X_train, y_train
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})
        temp = sess.run(cost, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        # 每次训练后输入在测试集上的准确率
        print(epoch, 'Accuracy:', acc)
    # 模型最后在测试集上的准确率
    print('final_arruracy:', sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0}))
    predictions = tf.arg_max(pred, 1)
    # 获取预测结果
    result = sess.run(predictions, feed_dict={x: X_test, keep_prob: 1.0})
    # f1值
    print(f1_score(y_test[:,1], result))