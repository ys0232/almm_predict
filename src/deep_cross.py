import numpy as np
import pandas as pd
import keras.backend as K
from keras import layers
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, Reshape, Add
from keras.layers import Flatten, concatenate, Lambda
from keras.models import Model
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import log_loss
import time
import gc
from sklearn.preprocessing import LabelEncoder

np.random.seed(1234)

def feature_generate(data):
    data, label, cate_columns, cont_columns = process_data(data)
    embeddings_tensors = []
    continuous_tensors = []
    for ec in cate_columns:
        layer_name = ec + '_inp'
        # For categorical features, we em-bed the features in dense vectors of dimension 6×(category cardinality)**(1/4)
        #nunique只作用于Series,用法是Series.nunique()，返回Series中只出现过一次的元素
        embed_dim = data[ec].nunique() if int(6 * np.power(data[ec].nunique(), 1 / 4)) > data[ec].nunique() \
            else int(6 * np.power(data[ec].nunique(), 1 / 4))
        #print(data[ec].nunique(),embed_dim)
        t_inp, t_build = embedding_input(layer_name, data[ec].nunique(), embed_dim)
        embeddings_tensors.append((t_inp, t_build))
        del (t_inp, t_build)
    for cc in cont_columns:
        layer_name = cc + '_in'
        t_inp, t_build = continous_input(layer_name)
        continuous_tensors.append((t_inp, t_build))
        del (t_inp, t_build)
    inp_layer = [et[0] for et in embeddings_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]
    inp_embed = [et[1] for et in embeddings_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]
    return data, label, inp_layer, inp_embed

def embedding_input(name, n_in, n_out):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1)(inp)

def continous_input(name):
    inp = Input(shape=(1,), dtype='float32', name=name)
    return inp, Reshape((1, 1))(inp)


# The optimal hyperparameter settings were 8 cross layers of size 54 and 6 deep layers of size 292 for DCN
# Embed "Soil_Type" column (embedding dim == 15), we have 8 cross layers of size 29
def fit(inp_layer, inp_embed, X, y, *params):  # X_val,y_val
    # inp_layer, inp_embed = feature_generate(X, cate_columns, cont_columns)
    input =concatenate (inp_embed,axis=-1)
    #print(input.shape)
    # deep layer
    for i in range(2):
        if i == 0:
            deep = Dense(8,activation='relu')(Flatten()(input))
        else:
            deep = Dense(8, activation='relu')(deep)
    # cross layer
    cross = CrossLayer(output_dim=input.shape[2].value, num_layer=2, name="cross_layer")(input)
    # concat both layers

    output = concatenate([deep, cross],axis=-1)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inp_layer, output)
    print(model.summary())
    # plot_model(model, to_file = 'model.png', show_shapes = True)
    model.compile(optimizer='Adamax', loss='binary_crossentropy')
    if len(params) == 2:
        X_val = params[0]
        y_val = params[1]
        hist=model.fit([X[c] for c in X.columns], y, batch_size=2000, epochs=10,shuffle=False,
                  validation_data=([X_val[c] for c in X_val.columns], y_val))
        print(hist.history['val_loss'])
        print(hist.history['loss'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    else:
        model.fit([X[c] for c in X.columns], y, batch_size=1024, shuffle=False,epochs=1)
    return model


# https://keras.io/layers/writing-your-own-keras-layers/
class CrossLayer(layers.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[2]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape=[1, self.input_dim], initializer='glorot_uniform', name='w_' + str(i),
                                          trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_' + str(i), trainable=True))
        self.built = True

    def call(self, input):
        for i in range(self.num_layer):
            if i == 0:
                cross = Lambda(lambda x: Add()(
                    [K.sum( K.batch_dot(x,K.reshape(x, (-1, self.input_dim, 1)))* self.W[i], 1, keepdims=True),
                     self.bias[i], x]))(input)
            else:
                cross = Lambda(lambda x: Add()(
                    [K.sum(K.batch_dot(input,K.reshape(x, (-1, self.input_dim, 1)))* self.W[i], 1, keepdims=True),
                     self.bias[i], input]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)


def process_data(data):
    lbl = LabelEncoder()
    # 处理离散特征
    data['item_id'] = lbl.fit_transform(data['item_id'])
    #lbl.fit_transform simliar to labelEncode
    data['user_id'] = lbl.fit_transform(data['user_id'])
    data['context_id'] = lbl.fit_transform(data['context_id'])
    data['shop_id'] = lbl.fit_transform(data['shop_id'])
    item_cat_1 = []  # 从属关系
    item_cat_2 = []
    for item in data['item_category_list']:
        item_cat_1.append(np.array(item.split(";")[1]))
        if(len(item.split(";"))>2):
            item_cat_2.append(np.array(item.split(";")[2]))
        else:
            item_cat_2.append(0)
    item_cat_1 = np.array(item_cat_1)
    item_cat_2 = np.array(item_cat_2)

    data['item_cat_1'] = item_cat_1
    data['item_cat_1'] = lbl.fit_transform(data['item_cat_1'])
    data['item_cat_2'] = item_cat_2
    data['item_cat_2'] = lbl.fit_transform(data['item_cat_2'])

    data['item_brand_id'] = lbl.fit_transform(data['item_brand_id'])
    data['item_city_id'] = lbl.fit_transform(data['item_city_id'])
    data['user_occupation_id'] = lbl.fit_transform(data['user_occupation_id'])
    data['context_page_id'] = lbl.fit_transform(data['context_page_id'])

    for i in range(2):
        data['item_property_list' + str(i)] = lbl.fit_transform(
            data['item_property_list'].map(
                lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for i in range(3):
        data['predict_category_' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i].split(':')[0]) if len(str(x).split(';')) > i else ''))
        for j in range(3):
            # print(j)
            data['predict_category_property_' + str(i) + '_' + str(j)] =lbl.fit_transform (data['predict_category_property'].map(
                lambda x: str(str(x).split(';')[i].split(':')[1].split(',')[j]) if len(str(x).split(';')) > i and
                                                                                   len(str(x).split(';')[i].split(':')) > 1 and
                                                                                   len(str(x).split(';')[i].split(':')[1].split(',')) > j else ''))
    # 时间特征
    hour = []
    day = []
    for item in data['context_timestamp']:
        value = time.localtime(item)
        hour.append(value.tm_hour)
        day.append(value.tm_mday)
    day = np.array(day)
    hour = np.array(hour)
    del data['context_timestamp']
    data['day'] = day
    data['hour'] = hour
    data['week'] = day
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_week_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_week_hour, 'left', on=['user_id', 'day', 'hour'])
    feat = ['user_id', 'shop_id', 'item_id','item_brand_id']
    for id in range(len(feat)):
        col = feat[id] + str('_times')
        id_cnt = data.groupby([feat[id]], as_index=False)['instance_id'].agg({col: 'count'})
        data = pd.merge(data, id_cnt, on=[feat[id]], how='left')

    del data['item_category_list']
    del data['predict_category_property']  # 先删除该特征，
    del data['item_property_list']  # 先删除该特征，
    scaler = StandardScaler()
    cont_columns = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                    'user_age_level', 'user_star_level', 'shop_review_num_level', 'shop_review_positive_rate','user_query_day',
                    'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'hour','week',
                    'user_query_day_hour','user_id_times','shop_id_times','item_id_times','item_brand_id_times'
                      ]

    cate_columns = [ 'item_cat_1', 'item_cat_2','predict_category_0','predict_category_1','item_property_list0','item_property_list1',
                    'predict_category_property_0_0','predict_category_property_0_1','item_brand_id',
                    'item_city_id','user_gender_id', 'user_occupation_id', 'context_page_id']
    label = data['is_trade']
    del data['is_trade']

    data_cont = pd.DataFrame(scaler.fit_transform(data[cont_columns]), columns=cont_columns)
    data_cont = data_cont.apply(lambda x: x.astype('float32'))
    data_cate = data[cate_columns]
    data = pd.concat([data_cate, data_cont, data.day], axis=1)
    return data, label, cate_columns, cont_columns

if __name__ == "__main__":
    data = pd.read_csv("../data/round2_train.txt", sep=" ")
    test = pd.read_csv("../data/round2_ijcai_18_test_b_20180510.txt", sep=" ")
    data = pd.concat([data, test], keys=['train', 'test'])
    gc.collect()
    X, y, inp_layer, inp_embed = feature_generate(data)
    del data
    gc.collect()

    online = False  # 这里用来标记是 线下验证 还是 在线提交
    if online == False:
        X = pd.concat([X, y], axis=1)
        print(X.info())
        X_train = X[(X['day']==31) | (X['day'] < 6)]  # 18,19,20,21,22,23,24
        X_test = X[(X['day'] == 6)]  # 暂时先使用第24天作为验证集
        #X_train = X[(X['day'] == 31) | (X['day'] < 24)]  # 18,19,20,21,22,23,24
        #X_test = X[(X['day'] == 24)]

        del X_train['day']
        del X_test['day']
        y_train = X_train.is_trade
        del X_train['is_trade']
        y_test = X_test.is_trade
        del X_test['is_trade']
    else:
        X = pd.concat([X, y], axis=1)
        print(X['day'].describe())
        del X['day']
        X_train = X[X.is_trade.notnull()]
        X_test = X[X.is_trade.isnull()]
        y_train = X_train.is_trade
        del X_train['is_trade']
        y_test = X_test.is_trade
        del X_test['is_trade']
    if online == False:
        model = fit(inp_layer, inp_embed, X_train, y_train, X_test, y_test)

        val_pre = model.predict([X_train[c] for c in X_train.columns], batch_size=2000)[:, 0]
        corr_pre = []
        for i in val_pre:
            if i == 0.0:
                corr_pre.append(10 ** -30)
            else:
                corr_pre.append(i)

        print("train log_loss",log_loss(y_train.values,corr_pre))
        val_pre = model.predict([X_test[c] for c in X_test.columns], batch_size=2000)[:, 0]
        corr_pre = []
        for i in val_pre:
            if i == 0.0:
                corr_pre.append(10 ** -30)
            else:
                corr_pre.append(i)

        print("test log_loss",log_loss(y_test.values,corr_pre))

    else:
        model = fit(inp_layer, inp_embed, X_train, y_train)
        val_pre = model.predict([X_test[c] for c in X_test.columns], batch_size=1024)[:, 0]
        corr_pre = []
        for i in val_pre:
            if i == 0.0:
                corr_pre.append(10 ** -30)
            else:
                corr_pre.append(i)
        test['predicted_score'] = corr_pre
        test[['instance_id', 'predicted_score']].to_csv('deep_cross_res_testb.txt', index=False, sep=' ')
        print('get it ! ')