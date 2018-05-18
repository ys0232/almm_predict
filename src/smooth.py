import scipy.special as special
from collections import Counter
import pandas as pd
import numpy
import random

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            print(new_alpha, new_beta, i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        #更新策略
        for i in range(iter_num):

            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

def smooth(data,cols):
    bs=BayesianSmoothing(1,1)
    train=data[~data['is_trade'].isnull()]
    traded=data[data['is_trade']==1]
    dic_i=dict(Counter(train[cols].values))
    dic_cov=dict(Counter(traded[cols].values))
    l=list(set(train[cols].values))
    I=[];C=[]
    for id in l:
        I.append(dic_i[id])
        if id not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[id])
    bs.update(I,C,10000,0.0001)
    return bs.alpha,bs.beta

if __name__ == '__main__':
    data = pd.read_csv("../data/round2_train.txt", sep="\s+")
    data = data.drop_duplicates(subset='instance_id')  # 把instance id去重
    print('make feature')
    bs=pd.DataFrame()
    for col in ['shop_id', 'item_id', 'user_id']:
        value = smooth(data, col)
        bs[col]=value
    print(bs.info())
    bs.to_csv('smooth.txt',encoding='utf-8',index=False)