import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import pandas as pd
import src.RankGauss
import time
from sklearn import preprocessing

def timestamp_datetime(value):
    # change timestamp to datetype data
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt
def getData(df_train):
    df_train = df_train.drop_duplicates(subset='instance_id')  # 把instance id去重
    lbl = preprocessing.LabelEncoder()
    df_train['len_item_category'] = df_train['item_category_list'].map(lambda x: len(str(x).split(';')))
    df_train['len_item_property'] = df_train['item_property_list'].map(lambda x: len(str(x).split(';')))
    for i in range(1, 3):
        df_train['item_category_list' + str(i)] = lbl.fit_transform(df_train['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
    for i in range(3):
        df_train['item_property_list' + str(i)] = lbl.fit_transform(
            df_train['item_property_list'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    feat=[ 'context_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id', 'user_occupation_id', 'user_gender_id']
    for col in feat:
        df_train[col] = df_train[col].apply(lambda x: 0 if x == -1 else x)
        df_train[col] = lbl.fit_transform(df_train[col])

    df_train['len_predict_category_property'] = df_train['predict_category_property'].map(lambda x: len(str(x).split(';')))
    for i in range(3):
        df_train['predict_category_' + str(i)] = lbl.fit_transform(df_train['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i].split(':')[0]) if len(str(x).split(';')) > i else ''))
        for j in range(3):
            df_train['predict_category_property_' + str(i) + '_' + str(j)] = lbl.fit_transform(df_train['predict_category_property'].map(
                lambda x: str(str(x).split(';')[i].split(':')[1].split(',')[j]) if len(str(x).split(';')) > i and
                                                                                   len(str(x).split(';')[i].split(':')) > 1 and
                                                                                   len(str(x).split(';')[i].split(':')[1].split(',')) > j else ''))
    df_train['realtime'] = df_train['context_timestamp'].apply(timestamp_datetime)
    df_train['realtime'] = pd.to_datetime(df_train['realtime'])
    df_train['day'] = df_train['realtime'].dt.day
    df_train['hour'] = df_train['realtime'].dt.hour
    feat = ['user_id', 'shop_id', 'item_id','item_brand_id', 'item_city_id','day','hour']
    df_train = df_train.sort_values(by=['context_timestamp', 'instance_id'])
    for id in feat:
        col = id + str('_times')
        id_cnt = df_train.groupby([id], as_index=False)['instance_id'].agg({col: 'count'})
        df_train = pd.merge(df_train, id_cnt, on=[id], how='left')
    df_train['is_new'] = df_train['user_id_times'].apply(lambda x: x if x < 2 else 0)
    feat=['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                'shop_review_num_level', 'shop_star_level','len_item_property','len_predict_category_property',
          'hour','pre_item','pre_shop','pre_context','item_category_list1','predict_category_0']
    for col in feat:
        means=np.mean(df_train[col])
        df_train[col] = df_train[col].apply(lambda x: means if x == -1 else x)
        df_train[col] = df_train[col].astype(float)
    return df_train
   # MODEL
def denoise_auto_encoder(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])

def DAE(data):
    col=[ 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level',
         'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'hour', 'day', 'pre_shop', 'user_id_times', 'pre_context',
         'shop_id_times', 'user_age_level_item_category_list1', 'shop_review_num_level_predict_category_0', 'item_id_times',
         'len_predict_category_property', 'shop_id','len_item_category']

    data.to_csv('input.csv', index=False)
    train_x = np.array(data[col].values, dtype=float)

    for i in range(train_x.shape[1]):
        # print('before RankGauss : ',train_x[:,i])
        train_x[:, i] = src.RankGauss.rank_gauss(train_x[:, i])
        # print('after RankGauss : ',train_x[:,i])
    print("data ready")
    # NETOWRK PARAMETERS
    print(train_x.shape)
    len0 = train_x.shape[1]
    n_input = len0
    n_hidden_1 = len0 // 2
    n_hidden_2 = len0 // 2
    n_output = len0
    print(n_input, n_hidden_1, n_hidden_2, n_output)

    # PLACEHOLDERS
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    dropout_keep_prob = tf.placeholder("float")

    # WEIGHTS
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }


    # MODEL AS A FUNCTION
    reconstruction = denoise_auto_encoder(x, weights, biases, dropout_keep_prob)
    print("NETOWRK READY")
    # COST
    cost = tf.reduce_mean(tf.pow(reconstruction - y, 2))
    # OPTIMIZER

    optm = tf.train.AdamOptimizer(0.01).minimize(cost)
    # INITIALIZER

    init = tf.global_variables_initializer()
    print("FUNCTIONS READY")

    savedir = "./"
    saver = tf.train.Saver(max_to_keep=1)
    print("SAVER READY")
    TRAIN_FLAG = 1
    epochs = 300
    batch_size = 4000
    disp_step = 1

    sess = tf.Session()
    sess.run(init)
    index = 0
    if TRAIN_FLAG:
        print("START OPTIMIZATION")
        for epoch in range(epochs):
            num_batch = int(train_x.shape[0] // batch_size)
            total_cost = 0.
            index = 0
            for i in range(num_batch):
                batch_xs = train_x[index:index + batch_size, :]
                batch_xs_noisy = batch_xs + 0.001 * np.random.randn(batch_size, len0)
                feeds = {x: batch_xs, y: batch_xs, dropout_keep_prob: 1.}
                sess.run(optm, feed_dict=feeds)
                total_cost += sess.run(cost, feed_dict=feeds)
                index += batch_size
            # DISPLAY
            if epoch % disp_step == 0:
                print("Epoch %02d/%02d average cost: %.6f"
                      % (epoch, epochs, total_cost / num_batch))
             # PLOT
                randidx = np.random.randint(train_x.shape[0], size=1)
                testvec = train_x[randidx, :]
            # print(testvec)
                noisyvec = testvec + 0.001 * np.random.randn(1, len0)
                outvec = sess.run(reconstruction, feed_dict={x: testvec, dropout_keep_prob: 1.})
                print('origin data: ', testvec)
                print('input data: ', noisyvec)
                print('output data: ', outvec)
                # outimg   = np.reshape(outvec, (28, 28))
            # SAVE
            # saver.save(sess, savedir + 'DAE.csv',global_step=epoch)
    outvec = sess.run(reconstruction, feed_dict={x: train_x, dropout_keep_prob: 1.})
    # print(outvec)
    # col.append('instance_id')
    out = pd.DataFrame(outvec, columns=col)
    out['instance_id'] = data['instance_id']
    out.to_csv('output.csv', index=False)
    # print(out.corr())
    print("OPTIMIZATION FINISHED")

if __name__=="__main__":
    train = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt", sep="\s+")
    testa = pd.read_csv("../data/round1_ijcai_18_test_a_20180301.txt", sep="\s+")
    data = pd.concat([train, testa])
    data = data.drop_duplicates(subset='instance_id')
    df_train = getData(data)
    # Denoising Autoencoders to preprocess data
    DAE(data)
