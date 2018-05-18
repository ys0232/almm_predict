import pandas as pd
import math

def getData(data):
    save = pd.DataFrame(
        columns=['instance_id', 'item_id', 'item_category_list', 'item_property_list', 'predict_category_property'])
    save['instance_id'] = data['instance_id']
    save['item_id'] = data['item_id']
    save['item_category_list'] = data['item_category_list']
    save['item_property_list'] = data['item_property_list']
    save['predict_category_property'] = data['predict_category_property']
    data = pd.read_csv('forTFIDF.csv', sep=' ')
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    print(data['len_item_property'].max())
    print(data['len_item_property'].describe())
    # print(data['len_item_property'].sort_values())

    cate_set = set();
    pred_cate = set()
    for i in range(1, 3):
        data['item_category_list' + str(i)] = data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else '')
        print('item_category_list', i, len(data['item_category_list' + str(i)].unique()))
        cate_set |= set(data['item_category_list' + str(i)].values)
        # if i!=1:
        #    print(data['item_category_list' + str(i)].unique(),i)
        # property_list = set()
    property_set = set();
    pred_property = set()
    for i in range(100):
        data['item_property_list' + str(i)] = data['item_property_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else '')
        print('item_property_list', i, len(data['item_property_list' + str(i)].unique()))
        # print(data['item_property_list' + str(i)].unique())
        property_set |= set(data['item_property_list' + str(i)].values)
    data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))
    print(data['len_predict_category_property'].max())
    max_cate_pre = data['len_predict_category_property'].max()
    for i in range(max_cate_pre):
        data['predict_category_' + str(i)] = data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i].split(':')[0]) if len(str(x).split(';')) > i else '')
        print('predict_category_', i, len(data['predict_category_' + str(i)].unique()))
        pred_cate |= set(data['predict_category_' + str(i)].values)
        data['len_pre_property_' + str(i)] = data['predict_category_property'].map(
            lambda x: len(str(str(x).split(';')[i].split(':')[1].split(';')))
            if len(str(x).split(';')) > i and len(str(x).split(';')[i].split(':')) > 1 else 0)
        print(data['len_pre_property_' + str(i)].max())
        print(data['len_pre_property_' + str(i)].describe())

        for j in range(data['len_pre_property_' + str(i)].max()):
            # print(j)
            data['predict_category_property_' + str(i)] = data['predict_category_property'].map(
                lambda x: str(str(x).split(';')[i].split(':')[1].split(',')[j])
                if len(str(x).split(';')) > i and len(str(x).split(';')[i].split(':')) > 1 and
                   len(str(x).split(';')[i].split(':')[1].split(',')) > j else '')
            print('predict_category_', i, j, len(data['predict_category_property_' + str(i)].unique()))
            if j > 20:
                print(data['predict_category_property_' + str(i)].unique())
            pred_property |= set(data['predict_category_property_' + str(i)].values)
    print(len(cate_set), len(pred_cate), len(pred_cate & cate_set))
    print(len(property_set), len(pred_property), len(pred_property & property_set))
    #data.to_csv('forTFIDF.csv', sep=' ',index=False)
    return data

def getCorpus(data):
    sim_cate = []
    corpus = {}
    cate_list = data['item_category_list'].values
    property_list = data['item_property_list'].values
    predict_cate_property = data['predict_category_property'].values
    for i in range(data.shape[0]):
        cate = str(cate_list[i]).split(';')
        if corpus.__contains__(cate[1]):
            property_dict = corpus[cate[1]]
        else:
            property_dict = {}
        property_ = str(property_list[i]).split(';')
        for j in range(len(property_)):
            if property_dict.__contains__(property_[j]):
                property_dict[property_[j]] += 1
            else:
                property_dict[property_[j]] = 1

        corpus[cate[1]] = property_dict
        if len(cate) > 2:
            corpus[cate[2]] = property_dict
    for i in corpus.keys():
        property_dict = corpus[i]
        # print(i)
        sum = 0
        size = corpus.__len__()
        for j in property_dict.keys():
            # tf
            sum += property_dict[j]
        # property_dict=property_dict.sort_values()
        # sorted(property_dict.items(), key=lambda d: d[1])
        for k in property_dict.keys():
            contain_pro = 0
            for cate_pro in corpus.keys():
                property_cate = corpus[cate_pro]
                if k in property_cate.keys():
                    # idf
                    contain_pro += 1

            property_dict[k] = property_dict[k] / sum * (math.log10(size / (1 + contain_pro)))
    for i in range(data.shape[0]):
        pred_cate_property = str(predict_cate_property[i]).split(';')
        cate = str(cate_list[i]).split(';')[1]
        property_dict = corpus[cate]
        cate_low = 0
        for pro in property_dict.keys():
            # fenmu1
            cate_low += property_dict[pro] * property_dict[pro]
        sim = 0
        for pred in pred_cate_property:
            sim_temp = 0
            # print(pred)
            pred_cate = str(pred).split(':')[0]
            if len(str(pred).split(':')) < 2:
                continue
            pred_property = str(pred).split(':')[1].split(',')
            for pred_pro in pred_property:
                if property_dict.__contains__(pred_pro):
                    # fenzi
                    sim_temp += property_dict[pred_pro]
            sim_temp = sim_temp / (math.sqrt(len(pred_property)) * math.sqrt(cate_low))
            if sim < sim_temp:
                sim = sim_temp
        print(i, sim)
        sim_cate.append(sim)
    data['category_sim'] = sim_cate
    return data


def getSim_property(data):
    # data = pd.read_csv('sim_process.csv', sep=' ')
    sim_property = []
    property_list = data['item_property_list'].values
    predict_cate_property = data['predict_category_property'].values
    for i in range(len(property_list)):
        property_item = str(property_list).split(';')
        item_pro_len = len(property_item)
        pred_cate_property = str(predict_cate_property[i]).split(';')
        sim = 0
        for pred in pred_cate_property:
            sim_temp = 0
            # print(pred)
            pred_cate = str(pred).split(':')[0]
            if len(str(pred).split(':')) < 2:
                continue
            pred_property = str(pred).split(':')[1].split(',')
            for pred_pro in pred_property:
                if property_item.__contains__(pred_pro):
                    # fenzi
                    sim_temp += 1
            sim_temp = sim_temp / (math.sqrt(len(pred_property)) * math.sqrt(item_pro_len))
            if sim < sim_temp:
                sim = sim_temp
        print(i, sim)
        sim_property.append(sim)
    temp = pd.read_csv('/home/yolin/tianchi/Advertising_transformation_prediction/processed.csv', sep=',')
    temp['property_sim'] = sim_property
    temp['category_sim'] = data['category_sim']
    temp.to_csv('/home/yolin/tianchi/Advertising_transformation_prediction/sim_processed.csv', sep=' ', index=False)


if __name__ == "__main__":
    train = pd.read_csv("../data/round1_ijcai_18_train_20180301.txt", sep="\s+")
    testa = pd.read_csv("../data/round1_ijcai_18_test_a_20180301.txt", sep="\s+")
    data = pd.concat([train, testa])
    data = data.drop_duplicates(subset='instance_id')
    data=getData(data)
    # process()
    data = getCorpus(data)
    getSim_property(data)
