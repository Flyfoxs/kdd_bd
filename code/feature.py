# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-04-17 19:32:26
  @Last Modified by:   zzn
  @Last Modified time: 2019-04-17 19:32:26
"""

import json
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import Counter

def read_profile_data():
    profile_data = pd.read_csv('../data/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data


def merge_raw_data():
    tr_queries = pd.read_csv('../data/train_queries.csv')
    te_queries = pd.read_csv('../data/test_queries.csv')
    tr_plans = pd.read_csv('../data/train_plans.csv')
    te_plans = pd.read_csv('../data/test_plans.csv')

    tr_click = pd.read_csv('../data/train_clicks.csv')

    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)

    te_data = te_queries.merge(te_plans, on='sid', how='left')
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def merge_split_data():
    te_queries = pd.read_csv('../data/test_queries.csv')
    te_plans = pd.read_csv('../data/test_plans.csv')

    te_data = te_queries.merge(te_plans, on='sid', how='left')
    te_data['click_mode'] = -1

    tr_data = pd.read_csv('../data/split_data.csv')

    data = pd.concat([tr_data, te_data], axis=0)
    data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def gen_od_feas(data):
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    return data


def gen_plan_feas(data):
    n = data.shape[0]
    mode_list_feas = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    mode_texts = []
    for i, plan in tqdm(enumerate(data['plans'].values)):
        try:
            cur_plan_list = json.loads(plan)
        except:
            cur_plan_list = []
        if len(cur_plan_list) == 0:
            mode_list_feas[i, 0] = 1
            first_mode[i] = 0

            max_dist[i] = -1
            min_dist[i] = -1
            mean_dist[i] = -1
            std_dist[i] = -1

            max_price[i] = -1
            min_price[i] = -1
            mean_price[i] = -1
            std_price[i] = -1

            max_eta[i] = -1
            min_eta[i] = -1
            mean_eta[i] = -1
            std_eta[i] = -1

            min_dist_mode[i] = -1
            max_dist_mode[i] = -1
            min_price_mode[i] = -1
            max_price_mode[i] = -1
            min_eta_mode[i] = -1
            max_eta_mode[i] = -1

            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            for tmp_dit in cur_plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            mode_list = np.array(mode_list, dtype='int')
            mode_list_feas[i, mode_list] = 1
            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)

            max_dist[i] = distance_list[distance_sort_idx[-1]]
            min_dist[i] = distance_list[distance_sort_idx[0]]
            mean_dist[i] = np.mean(distance_list)
            std_dist[i] = np.std(distance_list)

            max_price[i] = price_list[price_sort_idx[-1]]
            min_price[i] = price_list[price_sort_idx[0]]
            mean_price[i] = np.mean(price_list)
            std_price[i] = np.std(price_list)

            max_eta[i] = eta_list[eta_sort_idx[-1]]
            min_eta[i] = eta_list[eta_sort_idx[0]]
            mean_eta[i] = np.mean(eta_list)
            std_eta[i] = np.std(eta_list)

            first_mode[i] = mode_list[0]
            max_dist_mode[i] = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i] = mode_list[distance_sort_idx[0]]

            max_price_mode[i] = mode_list[price_sort_idx[-1]]
            min_price_mode[i] = mode_list[price_sort_idx[0]]

            max_eta_mode[i] = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i] = mode_list[eta_sort_idx[0]]

    feature_data = pd.DataFrame(mode_list_feas)
    feature_data.columns = ['mode_feas_{}'.format(i) for i in range(12)]
    feature_data['max_dist'] = max_dist
    feature_data['min_dist'] = min_dist
    feature_data['mean_dist'] = mean_dist
    feature_data['std_dist'] = std_dist

    feature_data['max_price'] = max_price
    feature_data['min_price'] = min_price
    feature_data['mean_price'] = mean_price
    feature_data['std_price'] = std_price

    feature_data['max_eta'] = max_eta
    feature_data['min_eta'] = min_eta
    feature_data['mean_eta'] = mean_eta
    feature_data['std_eta'] = std_eta

    feature_data['max_dist_mode'] = max_dist_mode
    feature_data['min_dist_mode'] = min_dist_mode
    feature_data['max_price_mode'] = max_price_mode
    feature_data['min_price_mode'] = min_price_mode
    feature_data['max_eta_mode'] = max_eta_mode
    feature_data['min_eta_mode'] = min_eta_mode
    feature_data['first_mode'] = first_mode
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]

    data = pd.concat([data, feature_data, mode_svd], axis=1)

    return data


def gen_profile_feas(data):
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1)
    data = data.merge(svd_feas, on='pid', how='left')
    return data


def gen_time_feas(data):
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    return data


def split_train_test(data):
    train_data = data[data['click_mode'] != -1]
    test_data = data[data['click_mode'] == -1]
    submit = test_data[['sid']].copy()
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, submit


def plan_mode_matrix(data):
    mode_list = []
    for value in data['plans'].values:
        try:
            mode_list.append(eval(value)[0]['transport_mode'])
        except:
            mode_list.append(-999)
    data['mode_rank'] = mode_list
    return data


def plan_speed_matrix(data):
    mode_list = []
    for value in data['plans'].values:
        try:
            mode_list.append(eval(value)[0]['distance']/float(eval(value)[0]['eta']))
        except:
            mode_list.append(-999)
    data['speed_rank'] = mode_list
    return data


# 计算出发点到目的地的角度方向,参考的是wiki的内容
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6378.137  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def arctan(data):
    arc_list = []
    for index in range(len(data)):
        arc_list.append(bearing_array(eval(data['o'][index])[0],
                                           eval(data['o'][index])[1],
                                           eval(data['d'][index])[0],
                                           eval(data['d'][index])[1]))

    data['arctan'] = arc_list
    return data





# 加入街道距离等.
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def Dummy_MahaDis(lat1, lng1, lat2, lng2):
    tmp1 = haversine_array(lat1, lng1, lat1, lng2)
    tmp2 = haversine_array(lat1, lng1, lat2, lng1)
    return tmp1 + tmp2

def distance(data):
    distance_list = []
    for index in range(len(data)):
        distance_list.append(Dummy_MahaDis(eval(data['o'][index])[0],
                                           eval(data['o'][index])[1],
                                           eval(data['d'][index])[0],
                                           eval(data['d'][index])[1]))
    data['distance'] = distance_list
    return data


#按时间段分段统计
def mode_tfidf(data):
    mode_list = []
    sid = data[['sid']]
    for plan_val in data['plans'].values:
        temp_str = ''
        try:
            for value in eval(plan_val):
                temp_str += 'mode{}'.format(str(value['transport_mode']))
                temp_str += " "
        except:
            temp_str += '0'
        mode_list.append(temp_str)

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_model = TfidfVectorizer(ngram_range=(1,2)).fit(mode_list)
    sparse_result = tfidf_model.transform(mode_list)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(sparse_result)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    mode_svd['sid'] = sid
    return mode_svd




# mode-distacne matrix
def mode_distance_svd(data):
    mode_distance_list = []
    for value in data['plans'].values:
        temp_dict = {}
        try:
            value = eval(value)
            for index in value:
                temp_dict[index['transport_mode']] = int(index['distance'])
        except:
            None
        mode_distance_list.append(temp_dict)
    temp_df = pd.DataFrame(mode_distance_list)
    temp_df.columns = ['mode_distance_svd_{}'.format(index) for index in range(11)]
    temp_df.fillna(0, inplace=True)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    distance_svd = svd_enc.fit_transform(temp_df.values)
    temp_df = pd.DataFrame(distance_svd)
    temp_df.columns = ['mode_distance_svd_{}'.format(index) for index in range(10)]
    data = pd.concat([data,temp_df],axis=1)
    return data

# mode-eta matrix
def mode_eta_svd(data):
    mode_eta_list = []
    for value in data['plans'].values:
        temp_dict = {}
        try:
            value = eval(value)
            for index in value:
                temp_dict[index['transport_mode']] = int(index['eta'])
        except:
            None
        mode_eta_list.append(temp_dict)
    temp_df = pd.DataFrame(mode_eta_list)
    temp_df.columns = ['mode_eta_svd_{}'.format(index) for index in range(11)]
    temp_df.fillna(0, inplace=True)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    eta_svd = svd_enc.fit_transform(temp_df.values)
    temp_df = pd.DataFrame(eta_svd)
    temp_df.columns = ['mode_eta_svd_{}'.format(index) for index in range(10)]
    data = pd.concat([data,temp_df],axis=1)
    return data


# mode-distacne matrix
def mode_speed_svd(data):
    mode_speed_list = []
    for value in data['plans'].values:
        temp_dict = {}
        try:
            value = eval(value)
            for index in value:
                temp_dict[index['transport_mode']] = int(index['distance'])/float(index['eta'])
        except:
            None
        mode_speed_list.append(temp_dict)
    temp_df = pd.DataFrame(mode_speed_list)
    temp_df.columns = ['mode_speed_svd_{}'.format(index) for index in range(11)]
    temp_df.fillna(0, inplace=True)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    speed_svd = svd_enc.fit_transform(temp_df.values)
    temp_df = pd.DataFrame(speed_svd)
    temp_df.columns = ['mode_speed_svd_{}'.format(index) for index in range(10)]
    data = pd.concat([data,temp_df],axis=1)
    return data

def rank_one_distance(data):
    distance_list = []
    for value in data['plans'].values:
        try:
            distance_list.append(eval(value)[0]['distance'])
        except:
            distance_list.append(0)
    data['rank_distance'] = distance_list
    return data

def rank_one_eta(data):
    eta_list = []
    for value in data['plans'].values:
        try:
            eta_list.append(eval(value)[0]['eta'])
        except:
            eta_list.append(0)
    data['rank_eta'] = eta_list
    return data

def rank_one_price(data):
    price_list = []
    for value in data['plans'].values:
        try:
            if eval(value)[0]['price'] == '':
                price_list.append(1)
            else:
                price_list.append(int(eval(value)[0]['price']))
        except:
            price_list.append(0)
    data['rank_price'] = price_list
    return data

def rank_div(data):
    data['distance_div_price'] = data['rank_distance'] / data['rank_price']
    data['eta_div_price'] = data['rank_eta'] / data['rank_price']
    return data

def rank_distance_sub(data):
    distance_sub_max = []
    distance_sub_mean = []
    distance_sub_median = []
    distance_sub_min = []
    distance_sub_quanti = []
    distance_sub_sum = []
    for value in data['plans'].values:
        try:
            distance = eval(value)[0]['distance']
            temp_distance_list = [val['distance'] for val in eval(value)]
            distance_sub_max.append(distance - np.max(temp_distance_list))
            distance_sub_mean.append(distance - np.mean(temp_distance_list))
            distance_sub_median.append(distance - np.median(temp_distance_list))
            distance_sub_min.append(distance - np.min(temp_distance_list))
            distance_sub_quanti.append(distance - np.percentile(temp_distance_list,q=75))
            distance_sub_sum.append(distance/np.sum(temp_distance_list))
        except:
            distance_sub_max.append(0)
            distance_sub_mean.append(0)
            distance_sub_median.append(0)
            distance_sub_min.append(0)
            distance_sub_quanti.append(0)
            distance_sub_sum.append(0)

    data['distance_sub_max'] = distance_sub_max
    data['distance_sub_mean'] = distance_sub_mean
    data['distance_sub_median'] = distance_sub_median
    data['distance_sub_min'] = distance_sub_min
    data['distance_sub_quanti'] = distance_sub_quanti
    data['distance_sub_sum'] = distance_sub_sum
    return data


def rank_eta_sub(data):
    eta_sub_max = []
    eta_sub_mean = []
    eta_sub_median = []
    eta_sub_min = []
    eta_sub_quanti = []
    eta_sub_sum = []
    for value in data['plans'].values:
        try:
            eta = eval(value)[0]['eta']
            temp_eta_list = [val['eta'] for val in eval(value)]
            eta_sub_max.append(eta - np.max(temp_eta_list))
            eta_sub_mean.append(eta - np.mean(temp_eta_list))
            eta_sub_median.append(eta - np.median(temp_eta_list))
            eta_sub_min.append(eta - np.min(temp_eta_list))
            eta_sub_quanti.append(eta - np.percentile(temp_eta_list,q=75))
            eta_sub_sum.append(eta/np.sum(temp_eta_list))
        except:
            eta_sub_max.append(0)
            eta_sub_mean.append(0)
            eta_sub_median.append(0)
            eta_sub_min.append(0)
            eta_sub_quanti.append(0)
            eta_sub_sum.append(0)

    data['eta_sub_max'] = eta_sub_max
    data['eta_sub_mean'] = eta_sub_mean
    data['eta_sub_median'] = eta_sub_median
    data['eta_sub_min'] = eta_sub_min
    data['eta_sub_quanti'] = eta_sub_quanti
    data['eta_sub_sum'] = eta_sub_sum
    return data



def rank_price_sub(data):
    price_sub_max = []
    price_sub_mean = []
    price_sub_median = []
    price_sub_min = []
    price_sub_quanti = []
    price_sub_sum = []
    for value in data['plans'].values:
        try:
            temp_price_list = [int(val['price']) if eval(val)[0]['price'] != '' else 1 for val in eval(value)]
            price = temp_price_list[0]
            price_sub_max.append(price - np.max(temp_price_list))
            price_sub_mean.append(price - np.mean(temp_price_list))
            price_sub_median.append(price - np.median(temp_price_list))
            price_sub_min.append(price - np.min(temp_price_list))
            price_sub_quanti.append(price - np.percentile(temp_price_list,q=75))
            price_sub_sum.append(price/np.sum(temp_price_list))
        except:
            price_sub_max.append(0)
            price_sub_mean.append(0)
            price_sub_median.append(0)
            price_sub_min.append(0)
            price_sub_quanti.append(0)
            price_sub_sum.append(0)

    data['price_sub_max'] = price_sub_max
    data['price_sub_mean'] = price_sub_mean
    data['price_sub_median'] = price_sub_median
    data['price_sub_min'] = price_sub_min
    data['price_sub_quanti'] = price_sub_quanti
    data['price_sub_sum'] = price_sub_sum
    return data




def get_train_test_feas_data():
    # data = merge_raw_data()
    data = merge_split_data()

    data = gen_od_feas(data)
    data = gen_plan_feas(data)
    data = gen_profile_feas(data)
    data = gen_time_feas(data)

    # data = plan_mode_matrix(data)
    data = arctan(data)
    data = distance(data)
    data = plan_speed_matrix(data)

    data = mode_distance_svd(data)
    data = mode_eta_svd(data)
    data = mode_speed_svd(data)


    data = rank_one_distance(data)
    data = rank_one_eta(data)
    data = rank_one_price(data)
    data = rank_div(data)
    data = rank_distance_sub(data)
    data = rank_eta_sub(data)
    data = rank_price_sub(data)


    data = data.drop(['o', 'd'], axis=1)
    data = data.drop(['plans'], axis=1)
    data.to_csv('../pre_data/data.csv',index=False)
    data = pd.read_csv('../pre_data/data.csv')
    train_x, train_y, test_x, submit = split_train_test(data)
    return train_x, train_y, test_x, submit

if __name__ == '__main__':
    pass
