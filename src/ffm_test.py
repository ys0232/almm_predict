import ffm
import pandas as pd
import numpy as np
import hashlib
from numba import jit
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class DF2libffm(object):
    def __init__(self, field_names):
        self.field_names = field_names

    def df2libffm(self, df):
        df_temp = pd.DataFrame()
        for row in df.values:
            features = []
            for field, feature in enumerate(self._gen_features(row)):
                features.append((field, self._hashstr(feature, 1e+6), 1))
            df_temp = pd.concat([df_temp, pd.DataFrame([features])])
        return df_temp

    def _gen_features(self, row):
        features = []
        for i, field in enumerate(self.field_names):
            value = row[i]
            key = field + '_' + str(value)
            features.append(key)
        return features

    def _hashstr(self, string, nr_bins=1e+6):
        return int(hashlib.md5(string.encode('utf8')).hexdigest(), 16) % (int(nr_bins) - 1) + 1

@jit
def df2libffm(df, field_Category=[], field_Numeric=[]):
        libffm = []
        num_n = len(field_Numeric)
        csr = OneHotEncoder().fit_transform(df[field_Category])  # csr_matrix，全名为Compressed Sparse Row
        if field_Category:
            for i in range(len(csr.indptr) - 1):
                ls = []
                k = csr.indptr[i + 1] - csr.indptr[i]
                for j in range(k):
                    ls.append((k - j + num_n, csr.indices[i * k + j] + num_n, 1))
                libffm.append(ls)

        if field_Numeric:
            for i in range(len(libffm)):
                for j in range(len(field_Numeric)):
                    libffm[i].append((j + 1, j + 1, df[field_Numeric[j]].iloc[i]))
        return np.array(libffm)




# prepare the data
# (field, index, value) format
def ffm_0():
    X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
         [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
         [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)], ]
    y = [1, 1, 0]
    ffm_data = ffm.FFMData(X, y)
    # train the model for 10 iterations
    n_iter = 10
    model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
    model.init_model(ffm_data)
    for i in range(n_iter):
        print('iteration %d, ' % i, end='')
    model.iteration(ffm_data)
    y_pred = model.predict(ffm_data)
    auc = log_loss(y, y_pred)
    print('train auc %.4f' % auc)



def ffm_test(ffmdata,data):
    # FFM
    X_train, X_test, y_train, y_test = train_test_split(ffmdata, data['is_trade'].values, test_size=0.3,
                                                        random_state=888)
    n_iter = 20
    ffm_train = ffm.FFMData(X_train, y_train)
    ffm_test = ffm.FFMData(X_test, y_test)
    model = ffm.FFM(eta=0.05, lam=0.01, k=10)
    model.init_model(ffm_train)
    for i in range(n_iter):
        model.iteration(ffm_train)
        y_true = model.predict(ffm_train)
        y_pred = model.predict(ffm_test)
        train_log = log_loss(y_train, y_true)
        test_log = log_loss(y_test, y_pred)
        print('iteration_%d: ' % i, 'train_auc %.4f' % train_log, 'test_auc %.4f' % test_log)

if __name__=="__main__":
    df_train = pd.read_csv('/home/yolin/tianchi/Advertising_transformation_prediction/src/DAE_mlp/output_process.csv',sep=',')
    col=['len_item_category', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level',
         'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'hour', 'day', 'pre_shop', 'user_id_times', 'pre_context',
         'shop_id_times', 'user_age_level_item_category_list1', 'shop_review_num_level_predict_category_0', 'item_id_times',
         'len_predict_category_property', 'shop_id', 'predict_category_property_0_0', 'item_id']

    ffmdata=df2libffm(df_train, col)
    ffm_test(ffmdata,df_train)