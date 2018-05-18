import warnings
from sklearn.metrics import log_loss
import lightgbm as lgb

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
from collections import Counter

def timestamp_datetime(value):
    # change timestamp to datetype data
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def find_pre(n):
    # to get some history features
    pre_dic = {}
    res = []
    for i in n.values:
        if i in pre_dic:
            res.append(pre_dic[i])
            pre_dic[i] = pre_dic[i] + 1
        else:
            res.append(0)
            pre_dic[i] = 1
    return np.array(res)

def smooth(data, cols, alpha, beta):
    # add BayesianSmooth feature
    cols_all = list(set(data[cols].values))
    train = data[~data['is_trade'].isnull()]
    traded = data[data['is_trade'] == 1]
    dic_i = dict(Counter(train[cols].values))
    dic_cov = dict(Counter(traded[cols].values))
    dic_PH = {}
    for id in cols_all:
        if id not in dic_i:
            dic_PH[id] = (alpha) / (alpha + beta)
        elif id not in dic_cov:
            dic_PH[id] = (alpha) / (dic_i[id] + alpha + beta)
        else:
            dic_PH[id] = (dic_cov[id] + alpha) / (dic_i[id] + alpha + beta)
    values = cols + '_trade_pro'
    df_out = pd.DataFrame({cols: list(dic_PH.keys()),
                           values: list(dic_PH.keys())})
    data = pd.merge(data, df_out, on=[cols], how='left')
    return data

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
    feat = ['context_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id', 'user_occupation_id',
            'user_gender_id']
    for col in feat:
        df_train[col] = df_train[col].apply(lambda x: 0 if x == -1 else x)
        df_train[col] = lbl.fit_transform(df_train[col])

    df_train['len_predict_category_property'] = df_train['predict_category_property'].map(
        lambda x: len(str(x).split(';')))
    for i in range(3):
        df_train['predict_category_' + str(i)] = lbl.fit_transform(df_train['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i].split(':')[0]) if len(str(x).split(';')) > i else ''))
        for j in range(3):
            df_train['predict_category_property_' + str(i) + '_' + str(j)] = lbl.fit_transform(
                df_train['predict_category_property'].map(
                    lambda x: str(str(x).split(';')[i].split(':')[1].split(',')[j]) if len(str(x).split(';')) > i and
                                                                                       len(str(x).split(';')[i].split(
                                                                                           ':')) > 1 and
                                                                                       len(str(x).split(';')[i].split(
                                                                                           ':')[1].split(
                                                                                           ',')) > j else ''))
    df_train['realtime'] = df_train['context_timestamp'].apply(timestamp_datetime)
    df_train['realtime'] = pd.to_datetime(df_train['realtime'])
    df_train['day'] = df_train['realtime'].dt.day
    df_train['hour'] = df_train['realtime'].dt.hour
    feat = ['user_id', 'shop_id', 'item_id', 'item_brand_id', 'item_city_id', 'day', 'hour']
    df_train = df_train.sort_values(by=['context_timestamp', 'instance_id'])
    for id in feat:
        col = id + str('_times')
        id_cnt = df_train.groupby([id], as_index=False)['instance_id'].agg({col: 'count'})
        df_train = pd.merge(df_train, id_cnt, on=[id], how='left')
    # get counts of different day features
    for d in range(19, 26):  # 18到24号
        df1 = df_train[df_train['day'] == d - 1]
        df2 = df_train[df_train['day'] == d]  # 19到25号
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id', 'day']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    df_train = pd.merge(df_train, Df2, on=['instance_id', 'day'], how='left')

    print('当前日期之前的cnt')
    for d in range(19, 26):
        # 19到25，25是test
        df1 = df_train[df_train['day'] < d]
        df2 = df_train[df_train['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cntx'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cntx'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id', 'day']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    df_train = pd.merge(df_train, Df2, on=['instance_id', 'day'], how='left')
    df_train = df_train.fillna(0)

    df_train['is_new'] = df_train['user_id_times'].apply(lambda x: x if x < 2 else 0)
    # to get some history features
    df_train['pre_item'] = df_train.groupby('user_id')['item_id'].transform(find_pre)
    df_train['pre_shop'] = df_train.groupby('user_id')['shop_id'].transform(find_pre)
    df_train['pre_context'] = df_train.groupby('user_id')['context_page_id'].transform(find_pre)
    df_train['pre_city'] = df_train.groupby('user_id')['item_city_id'].transform(find_pre)
    df_train['pre_brand'] = df_train.groupby('user_id')['item_brand_id'].transform(find_pre)

    feat = ['item_sales_level', 'item_price_level', 'item_collected_level',
            'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
            'shop_review_num_level', 'shop_star_level', 'len_item_property', 'len_predict_category_property',
            'hour', 'pre_item', 'pre_shop', 'pre_context', 'item_category_list1', 'predict_category_0']
    for col in feat:
        means = np.mean(df_train[col])
        df_train[col] = df_train[col].apply(lambda x: means if x == -1 else x)
        df_train[col] = df_train[col].astype(str)

        # to get some Combinatorial features
    df_train['user_age_level_item_category_list1'] = df_train['user_age_level'] + df_train['item_category_list1']
    df_train['user_age_level_item_category_list1'] = df_train['user_age_level_item_category_list1'].astype(float)
    df_train['shop_review_num_level_predict_category_0'] = df_train['shop_review_num_level'] + df_train[
        'predict_category_0']
    df_train['shop_review_num_level_predict_category_0'] = df_train['shop_review_num_level_predict_category_0'].astype(
        float)
    df_train['item_price_level_predict_category_0'] = df_train['item_price_level'] + df_train['predict_category_0']
    df_train['item_price_level_predict_category_0'] = df_train['item_price_level_predict_category_0'].astype(float)

    for col in feat:
        df_train[col] = df_train[col].astype(float)
    # get everyday user query times
    user_query_weekday = df_train.groupby(['user_id', 'day']).size().reset_index().rename(
        columns={0: 'user_query_day'})
    df_train = pd.merge(df_train, user_query_weekday, 'left', on=['user_id', 'day'])
    # get per hour user query times
    user_query_day_hour = df_train.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    df_train = pd.merge(df_train, user_query_day_hour, 'left',
                        on=['user_id', 'day', 'hour'])
    # use the parameters of BayesianSmooth to get features
    fw = open("/home/yolin/tianchi/Advertising_transformation_prediction/src/smooth.txt", 'r')
    cols_smooth = fw.readline().strip().split(',')
    value = [inst.strip().split(',') for inst in fw.readlines()]
    ph = pd.DataFrame(value, columns=cols_smooth)
    for col in ['shop_id', 'item_id', 'user_id']:
        alpha = float(ph[col].values[0])
        beta = float(ph[col].values[1])
        print(alpha, beta)
        df_train = smooth(df_train, col, alpha, beta)

    return df_train

def lgbClassify(df_train):
    train = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 23)]
    test = df_train[(df_train['day'] == 24)]
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'realtime', 'context_timestamp']]
    X = train[col]
    y = train['is_trade'].values
    x_test = test[col]
    y_test = test['is_trade'].values
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=12,
        learning_rate=0.01,
        max_depth=3,
        seed=2018,
        colsample_bytree=0.8,
        num_threads=8,
        subsample=0.9,
        n_estimators=8000)
    lgb_model = lgb0.fit(X, y, eval_set=[(x_test, y_test)], early_stopping_rounds=200)
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    print(feat_imp)
    print(feat_imp.shape)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['pred'] = pred
    test['index'] = range(len(test))
    print('误差 ', log_loss(test['is_trade'], test['pred']))
    return lgb_model.best_iteration_

def lgb_sub(train, test, best_iter):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'realtime', 'context_timestamp', 'user_gender_id', 'user_occupation_id']]
    X = train[col]
    y = train['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=12,
        num_threads=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        subsample=0.9,
        n_estimators=best_iter)
    lgb_model = lgb0.fit(X, y)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub = pd.read_csv("/home/yolin/tianchi/Advertising_transformation_prediction/round1_ijcai_18_test_b_20180418.txt",
                      sep="\s+")
    sub = pd.merge(sub, sub1, on=['instance_id'], how='left')
    sub = sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('result_12_1.txt', sep=" ", index=False)
if __name__ == "__main__":
    train = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt",sep="\s+")
    testa = pd.read_csv("../data/round1_ijcai_18_test_a_20180301.txt",sep="\s+")
    data = pd.concat([train, testa])
    data = data.drop_duplicates(subset='instance_id')
    df_train = getData(data)
    print('lightGBM result: ')
    best_tire=lgbClassify(df_train)
    train = df_train[df_train.is_trade.notnull()]
    test = df_train[df_train.is_trade.isnull()]
    # this is for submission
    lgb_sub(train,test,best_tire)
