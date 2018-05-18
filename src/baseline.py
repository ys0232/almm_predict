import warnings
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
from sklearn.linear_model import LogisticRegression

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
    sub=pd.read_csv("/home/yolin/tianchi/Advertising_transformation_prediction/round1_ijcai_18_test_b_20180418.txt", sep="\s+")
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('result_12_1.txt',sep=" ",index=False)
def FFMFormatPandas(pd_data):
    col_list = pd_data.columns
    field_index = dict(zip(col_list, range(len(col_list))))
    base_index = 0
    for col in pd_data.columns:
        if pd_data[col].dtype == 'object':
            vals = pd_data[col].unique()
            index_dict = dict(zip(vals, range(len(vals))))
            pd_data[col] = pd_data[col].map(lambda x: (field_index[col], base_index + index_dict[x], 1))
            base_index += len(vals)
        else:
            pd_data=pd_data.apply(lambda x: (x - x.mean()) / x.std())
            pd_data[col] = np.round(pd_data[col], 6)
            vals = pd_data[col].unique()
            index_dict = dict(zip(vals, range(len(vals))))
            pd_data[col] = pd_data[col].map(lambda x: (field_index[col], base_index, x))
            base_index += 1

    return pd_data.values
def logress(df_train):
    select_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                   'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_num_level',
                   'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
                   'shop_score_delivery', 'shop_score_description','property_sim','category_sim']
    df_train['len_item_property'] =df_train['item_property_list'].map(lambda x: len(str(x).split(';')))
    for col in ['user_occupation_id', 'user_gender_id']:
        # print(col)
        temp = pd.DataFrame()
        temp = pd.get_dummies(df_train[col], prefix=col)
        #print(type(temp),type(train_x))
        df_train = pd.concat([df_train, temp], axis=1)
    train=df_train[(df_train.day<24)]
    Y = train['is_trade']
    train=df_train.drop(df_train['is_trade'])
    train_x=train[select_cols]
    print('标准化。。。')
    train_x = train_x.apply(lambda x: (x - x.mean()) / x.std())  # 标准化
    model=LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(train_x, Y, test_size=0.4, random_state=0)
    print("Training...")
    model.fit(X_train, y_train)
    print("Predicting...")
    y_prediction = model.predict_proba(X_test)
    test_pred = y_prediction[:, 1]
    print('log_loss ', log_loss(y_test, test_pred))
def ffm(df_train, category_features):
    train = df_train[(df_train['day'] >= 18) & (df_train['day'] <= 23)]
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'realtime', 'context_timestamp']]
    raw_ffm_data = df_train
    for cols in category_features:
        raw_ffm_data[cols] = raw_ffm_data[cols].astype(str)
    data_ffm = FFMFormatPandas(raw_ffm_data[col])
    data_ffm_y = raw_ffm_data['is_trade'].tolist()
    X = train[col]
    train_num = X.shape[0]
    X_train_ffm = data_ffm[:train_num]
    X_test_ffm = data_ffm[train_num:]
    y_train_ffm = data_ffm_y[:train_num]
    y_test_ffm = data_ffm_y[train_num:]
    import ffm
    ffm_train = ffm.FFMData(X_train_ffm, y_train_ffm)
    ffm_test = ffm.FFMData(X_test_ffm, y_test_ffm)
    n_iter = 5
    ffmmodel = ffm.FFM(eta=0.02, lam=0.0001, k=6)
    ffmmodel.init_model(ffm_train)
    for i in range(n_iter):
        print('iteration %d : ' % i)
        ffmmodel.iteration(ffm_train)
        y_pred = ffmmodel.predict(ffm_test)
        t_pred = ffmmodel.predict(ffm_train)
        logloss = log_loss(y_test_ffm, y_pred)
        t_logloss = log_loss(y_train_ffm, t_pred)
        print('train log_loss %.4f' % (t_logloss), end='\t')
        print('test log_loss %.4f' % (logloss))

if __name__ == "__main__":
    train = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt",sep="\s+")
    testa = pd.read_csv("../data/round1_ijcai_18_test_a_20180301.txt",sep="\s+")
    data = pd.concat([train, testa])
    data = data.drop_duplicates(subset='instance_id')
    df_train = getData(data)
    # 1、lightGBM
    print('lightGBM result: ')
    best_tire=lgbClassify(df_train)
    train = df_train[df_train.is_trade.notnull()]
    test = df_train[df_train.is_trade.isnull()]
    # this is for submission
    lgb_sub(train,test,best_tire)
    # 2 、logistic regression
    print('logression result: ')
    logress(df_train)
    # 3、ffm
    print('ffm result: ')
    category_features = ['context_page_id', 'user_occupation_id', 'user_gender_id', 'context_id', 'item_id',
                         'user_id', 'shop_id', 'is_new']
    ffm(df_train, category_features)

