#!/apps/dslab/anaconda/python3/bin/python

import json
import time
import warnings
from functools import partial
from math import radians, atan, tan, sin, acos, cos, atan2, sqrt

import geohash
import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial.distance as dist

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame as DF
from scipy import stats
from six.moves import reduce
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from file_cache.utils.util_log import timed, timed_bolck, logger
from file_cache.cache import file_cache
from functools import lru_cache
warnings.filterwarnings('ignore')

def jsonLoads(strs, key):
    '''strs：传进来的json数据
       key：字典的键
    '''
    try:
        ret = []
        dict_ = json.loads(strs)
        for i in dict_:
            if i[key] == '':
                ret.append(0)
            else:
                ret.append(i[key])
        return ret
    except:
        return [-1]


def time_fun(x):
    try:
        return time.mktime(x.timetuple())
    except:
        return -1


def flatten_data(plans, col):
    """
    把plans  flatten
    """
    df = pd.DataFrame(list(plans[col].values))
    df['sid'] = plans['sid']
    dis = pd.DataFrame()
    for i in df.columns[:-1]:
        dis_df = df.loc[:, [i, 'sid']].copy()
        dis_df.columns = [col, 'sid']
        dis = pd.concat([dis, dis_df], axis=0, )
    dis = dis.dropna()
    #     dis = dis.sort_values('sid').reset_index(drop = True)
    return dis


def getDistance(latA, lonA, latB, lonB):  # 球面距离
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return np.nan


def bearing(lat1, lng1, lat2, lng2):  # 角度特征
    AVG_EARTH_RADIUS = 6378.137  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def get_city(local):  # 城市编码
    local = local.split(',')
    x = float(local[0])
    y = float(local[1])
    d1 = (x - 116.41) ** 2 + (y - 39.91) ** 2
    d2 = (x - 121.43) ** 2 + (y - 31.20) ** 2
    d3 = (x - 114.06) ** 2 + (y - 22.54) ** 2
    d4 = (x - 113.26) ** 2 + (y - 23.13) ** 2
    distance = [d1, d2, d3, d4]
    return np.argmin(distance)


def cal_manhattan_distance(O_lon, O_lat, D_lon, D_lat):  # 曼哈顿距离
    dlat = O_lat - D_lat
    a = sin(dlat / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    lat_d = c * r

    dlon = O_lat - D_lat
    a = sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    lon_d = c * r

    return lat_d + lon_d


base32 = {x: i + 1 for i, x in enumerate(list('0123456789bcdefghjkmnpqrstuvwxyz'))}  # GEOHASH 配置文件
def geohash2int(geohash_id):
    result = 0
    base = 1
    for each in geohash_id[::-1]:
        result = result + base32[each] * base
        base = base * 32
    return result


def get_ktime_feature(k, data, i):  # 排名为I时的特征
    kfc = data.copy()
    for time in range(1, k):
        tmp = kfc.sort_values(by=['sid', i]).drop_duplicates(subset=['sid'], keep='first')
        kfc = kfc[~kfc.index.isin(tmp.index)]
    tmp = kfc.sort_values(by=['sid', i]).drop_duplicates(subset=['sid'], keep='first')
    return tmp


def get_mode(x):  # 众数
    return stats.mode(x)[0][0]


def get_mode_count(x):  # 众数的统计值
    return stats.mode(x)[1][0]


# Graph Embedding
@timed()
def get_graph_embedding(data=None, cols=None, embed_size=128, isWeight=False, model_type=None, weight_col=[],
                        isGraph=False, intGraph=None):
    from ge import DeepWalk, Struc2Vec, SDNE, LINE, Node2Vec
    for i in tqdm([i for i in cols if i not in weight_col]):
        data[i] = data[i].astype('str')
    for i in weight_col:
        data[i] = data[i].astype('int')
    if isGraph:
        G = intGraph
    else:
        G = nx.DiGraph()
        if isWeight:
            G.add_weighted_edges_from(data[cols].drop_duplicates(subset=cols, keep='first').values)
        else:
            G.add_edges_from(data[cols].drop_duplicates(subset=cols, keep='first').values)
    if model_type == 'Node2Vec':
        model = Node2Vec(G, walk_length=10, num_walks=100, p=0.25, q=4, workers=1)  # init model
        model.train(embed_size=embed_size, window_size=5, iter=3)  # train model
    elif model_type == 'DeepWalk':
        model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)  # init model
        model.train(embed_size=embed_size, window_size=5, iter=3)  # train model
    elif model_type == "SDNE":
        model = SDNE(G, hidden_size=[256, 128])  # init model
        model.train(embed_size=embed_size, batch_size=3000, epochs=40, verbose=2)  # train model
    elif model_type == "LINE":
        model = LINE(G, embedding_size=128, order='second')  # init model,order can be ['first','second','all']
        model.train(embed_size=embed_size, batch_size=1024, epochs=50, verbose=2)  # train model
    elif model_type == "Struc2Vec":
        model = Struc2Vec(G, 10, 100, workers=1, verbose=40, )  # init model
        model.train(embed_size=embed_size, window_size=5, iter=3)  # train model

    embeddings = model.get_embeddings()  # get embedding vectors
    #     evaluate_embeddings(embeddings)
    embeddings = pd.DataFrame(embeddings).T
    new_col = "".join(cols)
    embeddings.columns = ['{}_{}_emb_{}'.format(new_col, model_type, i) for i in embeddings.columns]
    embeddings = embeddings.reset_index().rename(columns={'index': '{}'.format(cols[0])})

    return embeddings



def gen_pid_vec(x):
    """
    根据访问记录生成向量
    :param x:
    :return:
    """
    vec = np.zeros((24, 12))
    for v in x.values:
        vec[v[0], v[1]] = vec[v[0], v[1]] + 1
    return pd.Series(vec.reshape(24 * 12))

def gen_cluster(cluster_num, X):
    """
     # 将数据聚类
    :param cluster_num:
    :param X:
    :return:
    """
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    return y_pred, kmeans

def gen_od_vec_df(sample_od_df, fold, od_num_threshold):
    """
    生成访问向量，根据访问次数过滤
    :param pid_df:
    :return:
    """
    o_vec_columns = ['ov_f{}_{}'.format(fold, i) for i in range(24 * 12)]
    d_vec_columns = ['dv_f{}_{}'.format(fold, i) for i in range(24 * 12)]
    o_num_df = sample_od_df.groupby(['o'])[['hour', 'click_mode']].apply(gen_pid_vec).reset_index()
    o_num_df.columns = ['o'] + o_vec_columns
    o_num_df[o_vec_columns] = o_num_df[o_vec_columns].astype(np.float32)
    d_num_df = sample_od_df.groupby(['d'])[['hour', 'click_mode']].apply(gen_pid_vec).reset_index()
    d_num_df.columns = ['d'] + d_vec_columns
    d_num_df[d_vec_columns] = d_num_df[d_vec_columns].astype(np.float32)
    o_num_df['o_query_num'] = o_num_df[o_vec_columns].apply(lambda x: sum(x), axis=1).reset_index(drop=True)
    d_num_df['d_query_num'] = d_num_df[d_vec_columns].apply(lambda x: sum(x), axis=1).reset_index(drop=True)

    o_filtered_num_df = o_num_df[o_num_df['o_query_num'] >= od_num_threshold]
    d_filtered_num_df = d_num_df[d_num_df['d_query_num'] >= od_num_threshold]
    print('o less 500 o size: {}'.format(len(o_filtered_num_df)))

    print('d less 500 d size: {}'.format(len(d_filtered_num_df)))

    return o_filtered_num_df, d_filtered_num_df

def gen_multi_fold(od_df, o_vec_columns, d_vec_columns, fold, frac, od_num_threshold):
    """
    五折结果平均
    :param fold:
    :return:
    """
    ov_vec_list = []
    dv_vec_list = []
    # 五折随机抽样，统计 查询用户在不同时间，对不同模式的点击次数
    for fd in range(fold):
        # 只抽取 训练数据
        sample_od_df = od_df.loc[od_df['click_mode'] != -1].sample(frac=frac, random_state=2019)
        o_filtered_num_df, d_filtered_num_df = gen_od_vec_df(sample_od_df, fd, od_num_threshold)
        ov_vec_list.append(o_filtered_num_df)
        dv_vec_list.append(d_filtered_num_df)
    # 合并五折采样结果
    o_vec_merge_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['o'], how='left'), ov_vec_list)
    d_vec_merge_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['d'], how='left'), dv_vec_list)
    # 多折平均只值
    o_fold_df_list = []
    d_fold_df_list = []
    for f in range(fold):
        o_fold_columns = ['ov_f{}_{}'.format(f, i) for i in range(288)]
        d_fold_columns = ['dv_f{}_{}'.format(f, i) for i in range(288)]
        o_fold_df_list.append(o_vec_merge_df[o_fold_columns])
        d_fold_df_list.append(d_vec_merge_df[d_fold_columns])
    o_merge_folds_data = np.nanmean(np.stack(o_fold_df_list, axis=0), axis=0)
    d_merge_folds_data = np.nanmean(np.stack(d_fold_df_list, axis=0), axis=0)
    o_arr = o_vec_merge_df['o'].values[:, np.newaxis]
    d_arr = d_vec_merge_df['d'].values[:, np.newaxis]
    o_data = np.hstack([o_arr, o_merge_folds_data])
    d_data = np.hstack([d_arr, d_merge_folds_data])
    o_vec_df = pd.DataFrame(o_data)
    d_vec_df = pd.DataFrame(d_data)
    o_vec_df.columns = ['o'] + o_vec_columns
    d_vec_df.columns = ['d'] + d_vec_columns

    return o_vec_df, d_vec_df

def gen_od_svd_vec_df(o_vec_df, d_vec_df, dim, o_svd_columns, d_svd_columns):
    """

    :param pid_vec_df:
    :param fold:
    :return:
    """
    o_time_vec = o_vec_df[o_vec_df.columns[1:1 + 288]].values
    d_time_vec = d_vec_df[d_vec_df.columns[1:1 + 288]].values
    # svd 降维
    o_svd = TruncatedSVD(n_components=dim, n_iter=30, random_state=2019)
    o_svd_x = o_svd.fit_transform(o_time_vec)
    d_svd = TruncatedSVD(n_components=dim, n_iter=30, random_state=2019)
    d_svd_x = d_svd.fit_transform(d_time_vec)

    o_svd_vec_df = pd.DataFrame(np.hstack([o_vec_df.values[:, 0:1], o_svd_x]))
    d_svd_vec_df = pd.DataFrame(np.hstack([d_vec_df.values[:, 0:1], d_svd_x]))
    o_svd_vec_df.columns = ['o'] + o_svd_columns
    d_svd_vec_df.columns = ['d'] + d_svd_columns

    return o_svd_vec_df, d_svd_vec_df

def gen_od_vec_feats(base_featuers_df, folds, frac, od_num_threshold):
    """

    :param base_featuers_df:
    :param od_num_threshold:
    :return:
    """
    od_df = base_featuers_df[['sid', 'o', 'd', 'plan_time', 'click_mode', 'req_time_hour_0']]
    od_df.rename(columns={'req_time_hour_0' : 'hour'},inplace=True)
    od_df['click_mode'].fillna(0,inplace=True)
    od_df['click_mode'] = od_df['click_mode'].astype(np.int32)

    o_vec_columns = ['ov_{}'.format(i) for i in range(24 * 12)]
    d_vec_columns = ['dv_{}'.format(i) for i in range(24 * 12)]
    o_vec_df, d_vec_df = gen_multi_fold(od_df, o_vec_columns, d_vec_columns, folds, frac, od_num_threshold=od_num_threshold)
    o_svd_vec_list = []
    d_svd_vec_list = []
    for dim in [15, 20]:
        o_svd_columns = ['o_svd_d{}_{}'.format(dim, i) for i in range(dim)]
        d_svd_columns = ['d_svd_d{}_{}'.format(dim, i) for i in range(dim)]
        o_svd_vec_df, d_svd_vec_df = gen_od_svd_vec_df(o_vec_df, d_vec_df, dim, o_svd_columns, d_svd_columns)
        # 对svd向量聚类
        o_X = o_svd_vec_df[o_svd_columns]
        d_X = d_svd_vec_df[d_svd_columns]
        cluster_num_list = [10, 20, 30, 50]
        for cluster_num in cluster_num_list:
            o_svd_vec_df['o_vec_svd_cls_d{}_{}'.format(dim, cluster_num)], _ = gen_cluster(cluster_num=cluster_num, X=o_X)
            d_svd_vec_df['d_vec_svd_cls_d{}_{}'.format(dim, cluster_num)], _ = gen_cluster(cluster_num=cluster_num, X=d_X)
        o_svd_vec_list.append(o_svd_vec_df)
        d_svd_vec_list.append(d_svd_vec_df)
    o_svd_vec_merge = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['o'], how='left'), o_svd_vec_list)
    d_svd_vec_merge = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['d'], how='left'), d_svd_vec_list)

    merge_df1 = pd.merge(base_featuers_df[['sid', 'o', 'd']], o_svd_vec_merge, on=['o'], how='left')
    od_svd_vec_merge = pd.merge(merge_df1, d_svd_vec_merge, on=['d'], how='left')
    return od_svd_vec_merge


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def tokenize(data):
    tokenized_docs = [word_tokenize(doc) for doc in data]
    alpha_tokens = [[t.lower() for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [[lemmatizer.lemmatize(alpha) for alpha in doc] for doc in alpha_tokens]
    X_stem_as_string = [" ".join(x_t) for x_t in lem_tokens]
    return X_stem_as_string


def not_sid_col(x):
    return x[[i for i in x.columns if i not in ['sid']]]

def f1_macro(labels, preds):
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score, True

def get_f1_score(y, pred):
    pred_lst = pred.tolist()
    pred_lst = [item.index(max(item)) for item in pred_lst]
    score = []
    for i in range(12):
        score.append(f1_score([1 if i==item else 0 for item in y],
                              [1 if i==item else 0 for item in pred_lst]))
    c = Counter(y)
    score = [item*c[ix]/len(y) for ix, item in enumerate(score)]
    score = np.sum(score)
    print('f1-score = {:.4f}'.format(score))
    return score


# Plans Feature Distance
def get_stat(x):
    res = np.array([i for i in x if i != 0])
    if len(res) == 0:
        return 0
    else:
        return res


def edit_distance(word1, word2):
    try:
        len1 = len(word1);
        len2 = len(word2);
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i;
        for j in range(len2 + 1):
            dp[0][j] = j;

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return dp[len1][len2]
    except:
        return np.nan


def calc_distance(word1, word2, param='jaccard'):
    #      Param :
    #     'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    #     'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    #     'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
    #     'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    #     'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    try:
        matV = [word1, word2]
        return dist.pdist(matV, param)[0]
    except:
        return np.nan




# GLOBAL Param
cv = 5               # CV Folds, used in ratio and model train
random_seed = 2019   # Random Seed
for_test = True      # Control Test-Set
offline = False      # Use 11.24 - 11.31 For Offline Test
version = 2          # Phase

if for_test:
    nrows = None
else:
    nrows = 1000

input_dir = './input/data_set_phase{}/'.format(version)

t1 = time.time()
print("Now Input Data.....")

# if version == 2:
#     profiles = pd.read_csv(input_dir+'profiles.csv',nrows=nrows)
#     train_clicks_2 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version),parse_dates=['click_time'],nrows=nrows)
#     train_clicks_1 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version-1),parse_dates=['click_time'],nrows=nrows)
#     train_clicks = train_clicks_2.append(train_clicks_1).reset_index(drop=True)
#     train_plans_2 = pd.read_csv(input_dir+'train_plans_phase{}.csv'.format(version),parse_dates=['plan_time'],nrows=nrows)
#     train_plans_1 = pd.read_csv(input_dir+'train_plans_phase{}.csv'.format(version-1),parse_dates=['plan_time'],nrows=nrows)
#     train_plans = train_plans_2.append(train_plans_1).reset_index(drop=True)
#
#     test_plans = pd.read_csv(input_dir+'test_plans.csv',parse_dates=['plan_time'],nrows=nrows)
#
#     print("Use Time {}".format(time.time()-t1))
# else:
#     profiles = pd.read_csv(input_dir+'profiles.csv',nrows=nrows)
#     train_clicks = pd.read_csv(input_dir+'train_clicks.csv',parse_dates=['click_time'],nrows=nrows)
#     train_plans = pd.read_csv(input_dir+'train_plans.csv',parse_dates=['plan_time'],nrows=nrows)
#     train_queries = pd.read_csv(input_dir+'train_queries.csv',parse_dates=['req_time'],nrows=nrows)
#     test_plans = pd.read_csv(input_dir+'test_plans.csv',parse_dates=['plan_time'],nrows=nrows)
#     test_queries = pd.read_csv(input_dir+'test_queries.csv',parse_dates=['req_time'],nrows=nrows)
#     print("Use Time {}".format(time.time()-t1))

# print(profiles.shape,train_clicks.shape,train_plans.shape,train_queries.shape)
# print(test_plans.shape,test_queries.shape)

# if offline:
#     tmp = train_queries[train_queries['req_time']<'2018-12-01']
#     train_queries = tmp[tmp['req_time']<'2018-11-24']
#     test_queries = tmp[tmp['req_time']>='2018-11-24']
#     del tmp;

@timed()
@file_cache()
def get_queries():
    train_queries_2 = pd.read_csv(input_dir+'train_queries_phase{}.csv'.format(version),parse_dates=['req_time'],nrows=nrows)
    train_queries_1 = pd.read_csv(input_dir+'train_queries_phase{}.csv'.format(version-1),parse_dates=['req_time'],nrows=nrows)
    train_queries = train_queries_2.append(train_queries_1).reset_index(drop=True)

    test_queries = pd.read_csv(input_dir + 'test_queries.csv', parse_dates=['req_time'], nrows=nrows)

    train_queries['type_'] = 'train'
    test_queries['type_'] = 'test'

    #queries = pd.concat([train_queries.sample(frac=0.2), test_queries.sample(frac=0.2)], axis=0).reset_index(drop=True)
    queries = pd.concat([train_queries, test_queries], axis=0).reset_index(drop=True)
    return queries

@timed()
def get_train_clicks():
    train_clicks_2 = pd.read_csv(input_dir + 'train_clicks_phase{}.csv'.format(version), parse_dates=['click_time'],
                                 nrows=nrows)
    train_clicks_1 = pd.read_csv(input_dir + 'train_clicks_phase{}.csv'.format(version - 1), parse_dates=['click_time'],
                                 nrows=nrows)
    train_clicks = train_clicks_2.append(train_clicks_1).reset_index(drop=True)
    return train_clicks

def get_profiles():
    profiles = pd.read_csv(input_dir + 'profiles.csv')
    return profiles

@timed()
@lru_cache()
@file_cache()
def get_plans():

    print("Deal With Plans...")

    train_plans_2 = pd.read_csv(input_dir+'train_plans_phase{}.csv'.format(version),parse_dates=['plan_time'],nrows=nrows)
    train_plans_1 = pd.read_csv(input_dir+'train_plans_phase{}.csv'.format(version-1),parse_dates=['plan_time'],nrows=nrows)
    train_plans = train_plans_2.append(train_plans_1).reset_index(drop=True)
    test_plans = pd.read_csv(input_dir + 'test_plans.csv', parse_dates=['plan_time'], nrows=nrows)

    plans = pd.concat([train_plans,test_plans],axis=0).reset_index(drop = True)

    train_clicks = get_train_clicks()

    train_clicks.fillna(0,inplace=True)

    print(train_clicks.shape)

    queries = get_queries()

    plans = pd.merge(queries[['sid', 'type_']], plans, how='left', on='sid')
    plans = pd.merge(plans, train_clicks, how='left', on='sid')
    plans.loc[plans.type_=='test','click_mode'] = -1
    plans.click_mode = plans.click_mode.fillna(0)

    for i in tqdm(['distance','price','eta','transport_mode']):
        plans[i] = plans['plans'].apply(jsonLoads, key=i)

    with timed_bolck('plans_rank_att'):
        """transport_mode_rank"""
        plans['transport_mode_rank'] = plans['transport_mode'].apply(lambda x:np.arange(len(x)))
        plans['distance_rank'] = plans['distance'].apply(lambda x:np.argsort(x))
        plans['price_rank'] = plans['price'].apply(lambda x:np.argsort(x))
        plans['eta_rank'] = plans['eta'].apply(lambda x:np.argsort(x))
        plans['transport_mode_str'] = plans['transport_mode'].astype('str')
        plans['price_str'] = plans['price'].astype('str')
        plans['distance_str'] = plans['distance'].astype('str')
        plans['eta_str'] = plans['eta'].astype('str')

    def get_padding(x, delta=0):
        padding_maxlen = 8 # padding_maxlen = np.max(plans_feature['transport_mode_len'])
        if delta != 0:
            return list((x + delta)) + ([0] * (padding_maxlen - len(x)))
        else:
            return list((x)) + ([0] * (padding_maxlen - len(x)))

    with timed_bolck('plans_array_padding'):
        plans['distance_rank_array'] = plans['distance_rank'].map(lambda x: get_padding(x, 1))
        plans['eta_rank_array'] = plans['eta_rank'].map(lambda x: get_padding(x, 1))
        plans['price_rank_array'] = plans['price_rank'].map(lambda x: get_padding(x, 1))
        plans['mode_rank_array'] = plans['transport_mode_rank'].map(lambda x: get_padding(x, 1))

        plans['distance_array'] = plans['distance'].map(lambda x: get_padding(x, 0))
        plans['eta_array'] = plans['eta'].map(lambda x: get_padding(x, 0))
        plans['price_array'] = plans['price'].map(lambda x: get_padding(x, 0))
        plans['transport_mode_array'] = plans['transport_mode'].map(lambda x: get_padding(x, 0))


    plans = plans.sort_values(by=['sid'])
    return plans

@timed()
@lru_cache()
def get_plan_df():
    plans = get_plans()

    transport_mode_rank = flatten_data(plans, col = 'transport_mode_rank')
    distance = flatten_data(plans, col='distance', )
    eta = flatten_data(plans, col='eta')
    transport_mode = flatten_data(plans, col='transport_mode')
    price = flatten_data(plans, col='price')
    price.replace('', np.nan, inplace=True)

    plans_df = pd.concat([distance,transport_mode_rank.iloc[:,0],eta.iloc[:,0],transport_mode.iloc[:,0],price.iloc[:,0]],axis=1)

    transport_mode_list = plans[['sid','transport_mode']].copy()
    transport_mode_list.columns = ['sid','transport_mode_list']
    plans_df = plans_df.merge(plans[['sid','plan_time']], on='sid',how='left')

    return plans_df

@timed()
@lru_cache()
def get_plans_data():
    data = get_plans()
    data = data.loc[:,['sid','click_mode']]
    plans_df = get_plan_df()
    queries = get_queries()
    data = data.merge(plans_df, on='sid',how='left')
    data = data.merge(queries, on='sid',how='left')
    data['ep'] = data['eta'] / data['price'] # 单位时间所需价格
    data['dp'] = data['distance'] / data['price'] # 单位距离所需价格
    data['de'] = data['distance'] / data['eta'] # 单位距离所需时间
    data['ed'] = data['eta'] / data['distance'] # 单位eta所需距离
    data['pe'] = data['price'] / data['eta']
    data['pd'] = data['price'] / data['distance']

    data = data.sort_values(by=['sid'])

    print(data.shape,data.columns)
    print("Plans Prepare Finished...")
    return data

# # OD
# queries = get_queries()
# plans = get_plans()
# data  = get_plans_data()
# profiles = get_profiles()

#####  特征工程部分 #####

#Plan展开初级特征
@timed()
@file_cache()
def get_feature_plan_wide():
    data = get_plans_data()
    plans = get_plans()
    feature = plans[['sid']].sort_values(by=['sid']).copy()

    # 处理百度的推荐模型

    # 按顺序展开
    mixed_col = ['ep','pe','ed','de','dp','pd']
    for i in tqdm(range(0,4)):
        now = ['sid','distance','eta','price'] + mixed_col + ['transport_mode']
        tmp = data[data['transport_mode_rank']==i]
        tmp = tmp[now].set_index('sid').add_prefix("Recommand_{}_".format(i)).reset_index()
        feature = feature.merge(tmp,on='sid',how='left')

    # 在对应取得最小值/次小值时的对应推荐
    for i in tqdm(['distance','eta','price']+mixed_col):
        for j in range(1,3):
            tmp = get_ktime_feature(j,data,i)
            now = ['sid','distance','eta','price'] + mixed_col + ['transport_mode']
            now = [j for j in now if i not in j]
            tmp = tmp[now].set_index('sid').add_prefix("{}_inMin_{}_".format(i,j-1)).reset_index()
            feature = feature.merge(tmp,on='sid',how='left')
            if i in mixed_col:
                break
    return feature

@timed()
@file_cache()
def get_feature_from_plans():
    plans = get_plans()
    plans_feature = plans[['sid']]
    plans_feature['mode_array_count_sid'] = plans.groupby(['transport_mode_str'])['sid'].transform('count')
    plans_feature['price_count_sid'] = plans.groupby(['price_str'])['sid'].transform('count')
    plans_feature['eta_count_sid'] = plans.groupby(['eta_str'])['sid'].transform('count')
    plans_feature['distance_count_sid'] = plans.groupby(['distance_str'])['sid'].transform('count')
    plans_feature['mode_price_count'] = plans.groupby(['transport_mode_str','price_str'])['sid'].transform('count')
    plans_feature['mode_eta_count'] = plans.groupby(['transport_mode_str','eta_str'])['sid'].transform('count')
    plans_feature['mode_distance_count'] = plans.groupby(['transport_mode_str','distance_str'])['sid'].transform('count')

    plans_feature['transport_mode_len'] = plans['transport_mode'].map(lambda x:len(x))
    plans_feature['transport_mode_nunique'] = plans['transport_mode'].map(lambda x:len(set((x))))
    plans_feature['price_nonan_mean'] = plans['price'].map(lambda x:np.mean(get_stat(x)))
    # plans_feature['price_nonan_skew'] = plans['price'].map(lambda x:stats.skew(get_stat(x)))
    # plans_feature['price_nonan_kurt'] = plans['price'].map(lambda x:stats.kurtosis(get_stat(x)))
    plans_feature['price_nonan_std'] = plans['price'].map(lambda x:np.std(get_stat(x)))
    plans_feature['price_nonan_max'] = plans['price'].map(lambda x:np.max(get_stat(x)))
    plans_feature['price_nonan_min'] = plans['price'].map(lambda x:np.min(get_stat(x)))
    plans_feature['price_nonan_sum'] = plans['price'].map(lambda x:np.sum(get_stat(x)))

    padding_maxlen = np.max(plans_feature['transport_mode_len'])

    print('padding_maxlen=', padding_maxlen)


    plans_feature['price_have_0_num'] = plans['price'].map(lambda x:len([i for i in x if i==0]))
    plans_feature['price_have_0_ratio'] = plans_feature['price_have_0_num'] / plans_feature['transport_mode_len']
    plans_feature['price_mean'] = plans['price'].map(lambda x:np.mean(x))
    plans_feature['distance_mean'] = plans['distance'].map(lambda x:np.mean(x))
    plans_feature['eta_mean'] = plans['eta'].map(lambda x:np.mean(x))

    plans_feature['distance_min'] = plans['distance'].map(lambda x:np.min(x))
    plans_feature['distance_max'] = plans['distance'].map(lambda x:np.max(x))
    plans_feature['distance_std'] = plans['distance'].map(lambda x:np.std(x))
    plans_feature['distance_sum'] = plans['distance'].map(lambda x:np.sum(x))
    # plans_feature['distance_skew'] = plans['distance'].map(lambda x:stats.skew(x))
    # plans_feature['distance_kurt'] = plans['distance'].map(lambda x:stats.kurtosis(x))

    plans_feature['eta_min'] = plans['eta'].map(lambda x:np.min(x))
    plans_feature['eta_max'] = plans['eta'].map(lambda x:np.max(x))
    plans_feature['eta_std'] = plans['eta'].map(lambda x:np.std(x))
    plans_feature['eta_sum'] = plans['eta'].map(lambda x:np.sum(x))
    # plans_feature['eta_skew'] = plans['eta'].map(lambda x:stats.skew(x))
    # plans_feature['eta_kurt'] = plans['eta'].map(lambda x:stats.kurtosis(x))

    plans_feature['transport_mode_mode'] = plans['transport_mode'].map(lambda x:stats.mode(x))
    plans_feature['transport_mode_mode_count'] = plans_feature['transport_mode_mode'].map(lambda x:x[1][0])
    plans_feature['transport_mode_mode'] = plans_feature['transport_mode_mode'].map(lambda x:x[0][0])
    plans_feature['transport_mode_transform_count'] = plans_feature.groupby(['transport_mode_mode','transport_mode_mode_count'])['sid'].transform('count')



    print(plans.shape,plans.columns)
    for now in ['distance','price','eta']:
        emb1,emb2,emb3,emb4,emb5,emb6,emb7,emb8 = [],[],[],[],[],[],[],[]
        for i in tqdm(plans[['{}_array'.format(now),'{}_rank_array'.format(now),'mode_rank_array','transport_mode_array']].values):
            power1,power2 = [],[]
            for j in range(len(i[1])):
                power1.append(i[1][j]*3**(padding_maxlen-j-1))
                power2.append(i[2][j]*3**(padding_maxlen-j-1))
            emb1.append(np.dot(i[0],i[1]))
            emb2.append(np.dot(i[0],i[2]))
            emb3.append(np.dot(i[3],i[1]))
            emb4.append(np.dot(i[3],i[2]))
            emb5.append(np.dot(i[0],power1))
            emb6.append(np.dot(i[0],power2))
            emb7.append(np.dot(i[3],power1))
            emb8.append(np.dot(i[3],power2))

        plans_feature['{}_{}_itselfrank'.format(now,'dot')] = emb1
        plans_feature['{}_{}_moderank'.format(now,'dot')] = emb2
        plans_feature['{}_{}_his_mode'.format(now,'dot')] = emb3

        plans_feature['{}_{}_itselfrank'.format(now,'dot_power')] = emb5
        plans_feature['{}_{}_moderank'.format(now,'dot_power')] = emb6
        plans_feature['{}_{}_his_mode'.format(now,'dot_power')] = emb7

    plans_feature['transport_dot_rank'] = emb4
    plans_feature['transport_dot_power_rank'] = emb8

    distance_pair = [
        ('eta_array','price_array'),
        ('eta_array','distance_array'),
        ('price_array','distance_array'),
        ('eta_array','transport_mode_array'),
        ('eta_array','mode_rank_array'),
        ('price_array','transport_mode_array'),
        ('price_array','mode_rank_array'),
        ('distance_array','transport_mode_array'),
        ('distance_array','mode_rank_array'),
    ]

    for i in tqdm(distance_pair):
        plans_feature['{}_l2_distance'.format('_'.join(i))] = list(map(lambda x,y:calc_distance(x,y,'euclidean'),plans[i[0]],plans[i[1]]))
        plans_feature['{}_cos_distance'.format('_'.join(i))] = list(map(lambda x,y:calc_distance(x,y,'cosine'),plans[i[0]],plans[i[1]]))

    #     'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    #     'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
    #     'jaccard', 'jensenshannon', 'kulsinski', 'y7mahalanobis', 'matching',
    #     'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    #     'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

    plans_feature = plans_feature.sort_values(by=['sid']).reset_index(drop=True)
    return plans_feature

@timed()
@file_cache()
def get_feature_space_time():
    feature = get_feature_plan_wide()
    queries = get_queries()
    plans = get_plans()
    space_time = feature[['sid']].merge(queries,on=['sid'],how='left').merge(plans[['sid','plan_time']],on='sid',how='left')
    space_time['time_diff'] = ((space_time['req_time']-space_time['plan_time'])*1e-9).astype(int)
    space_time['req_time_dow'] = space_time['req_time'].dt.dayofweek
    space_time['req_time_woy'] = space_time['req_time'].dt.weekofyear
    space_time['req_is_weekend'] = (space_time['req_time'].dt.weekday>=5).astype(int)
    space_time['req_time_hour'] = space_time['req_time'].dt.hour+space_time['req_time'].dt.minute/60
    space_time['req_time_hour_0'] = space_time['req_time'].dt.hour

    for i in tqdm(['o','d']):
        space_time[i+'x'] = space_time[i].apply(lambda x:float(x.split(',')[0]))
        space_time[i+'y'] = space_time[i].apply(lambda x:float(x.split(',')[1]))

    city = []
    for i in tqdm(space_time['o']):
        city.append(get_city(i))

    space_time['city'] = city

    space_time['odl2_dis'] = np.sqrt((space_time['dx']-space_time['ox'])**2+(space_time['dy']-space_time['oy'])**2)
    space_time['dis_x'] = space_time['dx']-space_time['ox']
    space_time['dis_y'] = space_time['dy']-space_time['oy']
    sphere_dis = []
    deg = []
    mah = []
    for i in tqdm(space_time[['oy','ox','dy','dx']].values):
        sphere_dis.append(getDistance(i[0],i[1],i[2],i[3]))
        deg.append(bearing(i[0],i[1],i[2],i[3]))
        mah.append(cal_manhattan_distance(i[1],i[0],i[3],i[2]))

    space_time['deg'] = deg
    space_time['sphere_dis'] = sphere_dis
    space_time['mah'] = mah

    space_time['o_geohash'] = list(map(lambda x,y:geohash.encode(x,y,8),space_time['oy'],space_time['ox']))
    space_time['d_geohash'] = list(map(lambda x,y:geohash.encode(x,y,8),space_time['dy'],space_time['dx']))

    space_time['o_geohash'] = space_time['o_geohash'].map(geohash2int)
    space_time['d_geohash'] = space_time['d_geohash'].map(geohash2int)
    space_time.replace(np.inf,np.nan,inplace=True)
    print(space_time.shape,space_time.columns)

    # Graph Embedding Feature

    n2v = True
    s2v = False
    # 同构图
    # No Weight
    to_emb = ['o','d','req_time_hour_0','req_time_dow','ox','oy','dx','dy','pid']
    space_time_odh = space_time[to_emb+['sid']]
    space_time_odh['od'] = space_time_odh['o'] + ',' + space_time_odh['d']

    embedding_od_n2v = get_graph_embedding(data = space_time_odh, cols=['o','d'], embed_size=6, isWeight=False, model_type='Node2Vec')
    embedding_oxdx_n2v = get_graph_embedding(data = space_time_odh, cols=['ox','dx'], embed_size=6, isWeight=False, model_type='Node2Vec')
    embedding_oydy_n2v = get_graph_embedding(data = space_time_odh, cols=['oy','dy'], embed_size=6, isWeight=False, model_type='Node2Vec')

    space_time_odh['o_hour'] = space_time_odh['o'] + "_" + space_time_odh['req_time_hour_0'].astype('str')
    space_time_odh['d_hour'] = space_time_odh['d'] + "_" + space_time_odh['req_time_hour_0'].astype('str')

    space_time_odh['o_dow'] = space_time_odh['o'] + "_" + space_time_odh['req_time_dow'].astype('str')
    space_time_odh['d_dow'] = space_time_odh['d'] + "_" + space_time_odh['req_time_dow'].astype('str')

    space_time_odh['o_pid'] = space_time_odh['o'] + "_" + space_time_odh['pid'].astype('str')
    space_time_odh['d_pid'] = space_time_odh['d'] + "_" + space_time_odh['pid'].astype('str')

    embedding_od_hour_n2v = get_graph_embedding(data = space_time_odh, cols=['o_hour','d_hour'], embed_size=6, isWeight=False, model_type='Node2Vec')
    embedding_od_dow_n2v = get_graph_embedding(data = space_time_odh, cols=['o_dow','d_dow'], embed_size=6, isWeight=False, model_type='Node2Vec')
    embedding_od_pid_n2v = get_graph_embedding(data = space_time_odh, cols=['o_pid','d_pid'], embed_size=6, isWeight=False, model_type='Node2Vec')

    # Weight
    space_time_odh['weight'] = space_time_odh.groupby(['o'])['d'].transform('count')
    embedding_od_n2v_weight = get_graph_embedding(data = space_time_odh, cols=['o','d','weight'], embed_size=6, isWeight=True, weight_col=['weight'],model_type='Node2Vec')

    space_time_odh['weight'] = space_time_odh.groupby(['o_hour'])['d_hour'].transform('count')
    embedding_od_hour_n2v_weight = get_graph_embedding(data = space_time_odh, cols=['o_hour','d_hour','weight'], embed_size=6, isWeight=True, weight_col=['weight'],model_type='Node2Vec')

    space_time_odh['weight'] = space_time_odh.groupby(['o_dow'])['d_dow'].transform('count')
    embedding_od_dow_n2v_weight = get_graph_embedding(data = space_time_odh, cols=['o_dow','d_dow','weight'], embed_size=6, isWeight=True, weight_col=['weight'],model_type='Node2Vec')

    # 二分图 考虑Struc2Vec
    if s2v:
        embedding_od_s2v = get_graph_embedding(data = space_time_odh, cols=['o','d'], embed_size=4, isWeight=False, model_type='Struc2Vec')
        embedding_od_pid_s2v = get_graph_embedding(data = space_time_odh, cols=['od','pid'], embed_size=4, isWeight=False, model_type='Struc2Vec')
        embedding_hour_pid_s2v = get_graph_embedding(data = space_time_odh, cols=['pid','req_time_hour_0'], embed_size=4, isWeight=False, model_type='Struc2Vec')
        embedding_dow_pid_s2v = get_graph_embedding(data = space_time_odh, cols=['pid','req_time_dow'], embed_size=4, isWeight=False, model_type='Struc2Vec')
        embedding_opid_odow_s2v = get_graph_embedding(data = space_time_odh, cols=['o_pid','o_dow'], embed_size=4, isWeight=False, model_type='Struc2Vec')

    # Merge Feature
    space_time_odh = space_time_odh.merge(embedding_od_n2v,how='left',on=['o']).\
                                    merge(embedding_oxdx_n2v,how='left',on=['ox']).\
                                    merge(embedding_oydy_n2v,how='left',on=['oy']).\
                                    merge(embedding_od_hour_n2v,how='left',on=['o_hour']).\
                                    merge(embedding_od_dow_n2v,how='left',on=['o_dow']).\
                                    merge(embedding_od_pid_n2v,how='left',on=['o_pid']).\
                                    merge(embedding_od_n2v_weight,how='left',on=['o']).\
                                    merge(embedding_od_hour_n2v_weight,how='left',on=['o_hour']).\
                                    merge(embedding_od_dow_n2v_weight,how='left',on=['o_dow'])
    if s2v:
        space_time_odh = space_time_odh.merge(embedding_od_s2v,how='left',on=['o']).\
                                        merge(embedding_od_pid_s2v,how='left',on=['od']).\
                                        merge(embedding_hour_pid_s2v,how='left',on=['pid']).\
                                        merge(embedding_dow_pid_s2v,how='left',on=['pid']).\
                                        merge(embedding_opid_odow_s2v,how='left',on=['o_pid'])

    print(space_time_odh.shape,space_time_odh.columns)
    space_time_odh = space_time_odh[['sid'] + [i for i in space_time_odh.columns if 'emb' in i]]
    print(space_time_odh.shape,space_time_odh.columns)


    # Count Feature
    space_time['sphere_dis_bins'] = pd.cut(space_time['sphere_dis'],bins=20)
    to_group = [
        'pid','o','d','oy','ox','dy','dx',
        'req_time_dow','req_is_weekend','req_time_hour','sphere_dis_bins',#'Recommand_0_price_bins',
        #'Recommand_0_transport_mode','Recommand_1_transport_mode','Recommand_2_transport_mode','price_inMin_0_transport_mode'
    ]

    gen_1,gen_2,gen_3,gen_4 = [],[],[],[]
    for i in tqdm(range(len(to_group))):
        for j in range(i+1,len(to_group)):
            gen_1.append([to_group[i],to_group[j]])
            for k in range(j+1,len(to_group)):
                for m in range(k+1,len(to_group)):
                    gen_4.append([to_group[i],to_group[j],to_group[k],to_group[m]])
                gen_3.append([to_group[i],to_group[j],to_group[k]])
                gen_2.append(([to_group[i],to_group[j]],to_group[k]))
    print(len(gen_1),len(gen_2),len(gen_3),len(gen_4))

    agg_count_3 = space_time[to_group+['sid']]

    for i in tqdm(gen_3):
        if ('_'.join(i)+'_agg_count' not in agg_count_3.columns):
            agg_count_3['_'.join(i)+'_agg_count'] = agg_count_3[i+['sid']].groupby(i)['sid'].transform('count')

    agg_count_3 = agg_count_3[[i for i in agg_count_3.columns if i not in ['sid','click_mode']+to_group]]

    print("Before Merge: ",space_time.shape)
    space_time = pd.concat([space_time,agg_count_3],axis=1)
    print(space_time.shape,agg_count_3.shape)#,agg_count_4.shape
    return space_time


@timed()
@file_cache()
def get_feature_od_svd_vec():
    space_time = get_feature_space_time()
    train_clicks = get_train_clicks()
    od_num_threshold = 30
    frac = 0.8
    od_svd_vec = gen_od_vec_feats(space_time.merge(train_clicks[['sid','click_mode']],how='left',on='sid'),
                                  cv, frac, od_num_threshold)
    od_svd_vec = od_svd_vec[[i for i in od_svd_vec.columns if 'svd' in i] + ['sid']]
    for i in tqdm(od_svd_vec.columns):
        od_svd_vec[i] = od_svd_vec[i].astype(np.float16)
    return od_svd_vec

@timed()
@file_cache()
def get_feature_build() :
    feature = get_feature_plan_wide()
    r0 = ['Recommand_0_{}'.format(i) for i in ['eta','distance','price']]
    r1 = ['Recommand_1_{}'.format(i) for i in ['eta','distance','price']]
    r2 = ['Recommand_2_{}'.format(i) for i in ['eta','distance','price']]
    dis = [
        (feature[r0].fillna(0).values,feature[r1].fillna(0).values),
        (feature[r0].fillna(0).values,feature[r2].fillna(0).values),
        (feature[r1].fillna(0).values,feature[r2].fillna(0).values),
    ]

    tmp0,tmp1 = [],[]
    for i in dis:
        col0,col1 = [],[]
        for j in tqdm(range(len(i[0]))):
            col0.append(calc_distance(i[0][j],i[1][j],'cosine'))
            col1.append(calc_distance(i[0][j],i[1][j],'euclidean'))
        tmp0.append(col0)
        tmp1.append(col1)

    tmp0 = pd.DataFrame(np.array(tmp0).T)
    tmp0.columns = ['Recommand_0_1_cos','Recommand_0_2_cos','Recommand_1_2_cos',]
    to_calc = list(tmp0.columns)
    tmp0['012_cos_mean'] = tmp0[to_calc].mean(axis=1)
    tmp0['012_cos_sum'] = tmp0[to_calc].sum(axis=1)
    tmp0['012_cos_std'] = tmp0[to_calc].std(axis=1)

    tmp1 = pd.DataFrame(np.array(tmp1).T)
    tmp1.columns = ['Recommand_0_1_l2','Recommand_0_2_l2','Recommand_1_2_l2']
    to_calc = list(tmp1.columns)

    tmp1['012_l2_mean'] = tmp1[to_calc].mean(axis=1)
    tmp1['012_l2_sum'] = tmp1[to_calc].sum(axis=1)
    tmp1['012_l2_std'] = tmp1[to_calc].std(axis=1)

    feature = pd.concat([feature, tmp0, tmp1],axis=1)
    print(feature.shape)

    # Sequence 2 Sequence Graph Embedding
    space_time = get_feature_space_time()
    plans = get_plans()
    to_build = [
        space_time[['sid','o','d','pid','req_time_hour_0','req_time_dow']],
        feature[[i for i in feature.columns if (('Recommand_' in i) | ('inMin' in i)) & ('transport_mode' in i)]],
        plans[[i for i in plans.columns if 'array' in i] + ['transport_mode_rank']]
    ]
    to_build = pd.concat(to_build,axis=1)
    to_build['od'] = to_build['o'] + to_build['d']
    to_build['od_pid'] = to_build['od'] + to_build['pid'].astype('str')
    to_build['price_distance_eta_inMin0'] = to_build['price_inMin_0_transport_mode'].astype('str') + to_build['eta_inMin_0_transport_mode'].astype('str') + to_build['distance_inMin_0_transport_mode'].astype('str')

    to_build_col = [i for i in to_build.columns if i not in ['sid']]

    embedding_1 = get_graph_embedding(data = to_build, cols=['od_pid','transport_mode_array'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_2 = get_graph_embedding(data = to_build, cols=['od_pid','transport_mode_rank'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_3 = get_graph_embedding(data = to_build, cols=['transport_mode_array','transport_mode_rank'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_4 = get_graph_embedding(data = to_build, cols=['od','transport_mode_rank'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_5 = get_graph_embedding(data = to_build, cols=['od','transport_mode_array'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_6 = get_graph_embedding(data = to_build, cols=['od','price_distance_eta_inMin0'], embed_size=4, isWeight=False, model_type='Node2Vec')

    embedding_7 = get_graph_embedding(data = to_build, cols=['pid','transport_mode_array'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_8 = get_graph_embedding(data = to_build, cols=['pid','price_distance_eta_inMin0'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_9 = get_graph_embedding(data = to_build, cols=['pid','price_rank_array'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_10 = get_graph_embedding(data = to_build, cols=['pid','distance_rank_array'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_11 = get_graph_embedding(data = to_build, cols=['pid','eta_rank_array'], embed_size=4, isWeight=False, model_type='Node2Vec')
    embedding_12 = get_graph_embedding(data = to_build, cols=['price_distance_eta_inMin0','transport_mode_array'], embed_size=4, isWeight=False, model_type='Node2Vec')

    to_build = to_build.merge(embedding_1,how='left',on=['od_pid']).\
                        merge(embedding_2,how='left',on=['od_pid']).\
                        merge(embedding_3,how='left',on=['transport_mode_array']).\
                        merge(embedding_4,how='left',on=['od']).\
                        merge(embedding_5,how='left',on=['od']).\
                        merge(embedding_6,how='left',on=['od']).\
                        merge(embedding_7,how='left',on=['pid']).\
                        merge(embedding_8,how='left',on=['pid']).\
                        merge(embedding_9,how='left',on=['pid']).\
                        merge(embedding_10,how='left',on=['pid']).\
                        merge(embedding_11,how='left',on=['pid']).\
                        merge(embedding_12,how='left',on=['price_distance_eta_inMin0'])

    to_build = to_build[[i for i in to_build.columns if i not in to_build_col]].sort_values(by=['sid'],ascending=True)

    return to_build

@timed()
@file_cache()
def get_feature_txt():
    plans = get_plans()
    # Text
    N_COM = 5

    vct = CountVectorizer(stop_words='english',analyzer='char_wb', lowercase=False,min_df=2)
    svd = TruncatedSVD(n_components=N_COM, random_state=2019)
    tfvec = TfidfVectorizer(ngram_range=(1, 6),analyzer='char_wb')
    text_feature = plans[['sid']].sort_values(by=['sid'])

    for i in tqdm(['distance','price','eta','transport_mode']):
        text = plans[i].astype(str).fillna('NAN').values
        x = tfvec.fit_transform(text)
        x = svd.fit_transform(x)
        svd_feas = pd.DataFrame(x)
        svd_feas.columns = ['{}_svd_tfidf_fea_{}'.format(i,j) for j in range(N_COM)]
        svd_feas['sid'] = plans['sid'].values
        text_feature = text_feature.merge(svd_feas, on='sid', how='left')

        x = vct.fit_transform(text)
        x = svd.fit_transform(x)
        svd_feas = pd.DataFrame(x)
        svd_feas.columns = ['{}_svd_countvec_fea_{}'.format(i,j) for j in range(N_COM)]
        svd_feas['sid'] = plans['sid'].values
        text_feature = text_feature.merge(svd_feas, on='sid', how='left')

    text_feature = text_feature.sort_values(by=['sid'])
    print(text_feature.shape)
    return text_feature


@timed()
@file_cache()
def get_feature_pid():
    profiles = get_profiles().copy()
    queries = get_queries()
    data = get_plans_data()
    feature = get_feature_plan_wide()
    pid_stats = feature[['sid']].merge(queries[['sid','pid']],how='left',on='sid')
    pid_stats = pid_stats.merge(profiles,on='pid',how='left')
    print(pid_stats.shape)

    tmp = data[['pid','transport_mode']].groupby(['pid'])['transport_mode'].agg(['median','std','nunique','count',get_mode,get_mode_count]).add_prefix('pid_transport_mode_').reset_index()
    pid_stats = pid_stats.merge(tmp,how='left',on='pid')

    N_COM = 5
    x = profiles.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=N_COM, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_pid_fea_{}'.format(i) for i in range(N_COM)]
    svd_feas['pid'] = profiles['pid'].values
    pid_stats['pid'] = pid_stats['pid'].fillna(-1)
    pid_stats = pid_stats.merge(svd_feas, on='pid', how='left')
    print(pid_stats.shape)
    pid_stats['isnull_sum_pid'] = pid_stats[pid_stats.columns].isnull().sum(axis=1)

    for i in tqdm(['price','eta','distance']):
        tmp = data[['pid',i]].groupby(['pid'])[i].agg(['std','min','max','mean']).add_prefix('pid_{}_'.format(i)).reset_index()
        pid_stats = pid_stats.merge(tmp,how='left',on='pid')

    del pid_stats['pid']

    return pid_stats

def get_feature_name():
    feature_name = [i for i in all_data.columns if i not in ['sid','click_mode','plan_time','req_time','label', 'type_']]
    return feature_name

@timed()
def get_feature_all():
    pid_stats     = get_feature_pid()
    feature       = get_feature_plan_wide()
    plans_feature = get_feature_from_plans()
    text_feature  = get_feature_txt()


    #Embedding model 耗时比较久
    # to_build      = get_feature_build()
    #space_time    = get_feature_space_time()
    #od_svd_vec = get_feature_od_svd_vec()


    all_data = pd.concat([pid_stats,
                          not_sid_col(feature),
                          not_sid_col(plans_feature),
                          not_sid_col(text_feature),

                          # not_sid_col(to_build),
                          # not_sid_col(space_time),
                          #not_sid_col(od_svd_vec),

                         ],axis=1)

    train_clicks = get_train_clicks()
    queries = get_queries()

    all_data = all_data.merge(train_clicks[['sid','click_mode']],how='left',on='sid')
    train = queries.loc[queries.type_ == 'train']
    all_data.loc[(all_data.sid.isin(train.sid)) & pd.isna(all_data.click_mode), 'click_mode']=0

    print(all_data.shape,all_data.columns)

    from sklearn.preprocessing import LabelEncoder
    cate_feature = ['oy','ox','dx','dy','pid','p0','o','d','o_geohash','d_geohash','req_time_dow','req_is_weekend','sphere_dis_bins']

    with timed_bolck('LabelEncoder'):
        for i in tqdm(cate_feature):
            try:
                lbl = LabelEncoder()
                all_data[i] = lbl.fit_transform(all_data[i].astype('str'))
            except:
                logger.exception(i)
                continue

    print(len(cate_feature),)
    return all_data

if __name__ == '__main__':

    """
    运行方式:
    nohup python ph3/kdd_phase3_refactor.py &
    
    快速测试代码逻辑错: 
    get_queries,里面的采样比例即可
    
    """



    all_data = get_feature_all()
    # Define F1 Train

    # CV TRAIN
    from collections import Counter

    feature_name = get_feature_name()
    tr_index = ~all_data['click_mode'].isnull()
    X_train = all_data[tr_index][list(set(feature_name))].reset_index(drop=True)
    y = all_data[tr_index]['click_mode'].reset_index(drop=True)
    X_test = all_data[~tr_index][list(set(feature_name))].reset_index(drop=True)
    print(X_train.shape,X_test.shape)
    final_pred = []
    cv_score = []
    cv_model = []
    skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        with timed_bolck(f'CV_Folder#{index}'):
            lgb_model = lgb.LGBMClassifier(
                boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
                max_depth=-1, n_estimators=3000, objective='multiclass',num_classes=12,
                subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
                learning_rate=0.1, random_state=2019 + index, n_jobs=40, metric="None", importance_type='gain'
            )
            train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            eval_set = [(test_x[feature_name], test_y)]
            lgb_model.fit(train_x[feature_name], train_y, eval_set=eval_set,verbose=10,early_stopping_rounds=30,eval_metric=f1_macro)
            cv_model.append(lgb_model)
            y_test = lgb_model.predict(X_test[feature_name])
            y_val = lgb_model.predict_proba(test_x[feature_name])
            print(Counter(np.argmax(y_val,axis=1)))
            cv_score.append(get_f1_score(test_y,y_val))
            if index == 0:
                final_pred = np.array(y_test).reshape(-1, 1)
            else:
                final_pred = np.hstack((final_pred, np.array(y_test).reshape(-1, 1)))
#
# import matplotlib.pyplot as plt
#
# fi = []
# for i in cv_model:
#     tmp = {
#         'name' : feature_name,
#         'score' : i.feature_importances_
#     }
#     fi.append(pd.DataFrame(tmp))
#
# fi = pd.concat(fi)
# fig = plt.figure(figsize=(8,5))
# # fi.groupby(['name'])['score'].agg('sum').sort_values(ascending=False).head(20).plot.barh()
# fi.groupby(['name'])['score'].agg('mean').sort_values(ascending=False).head(30).plot.barh()
#
# cv_pred = np.zeros((X_train.shape[0],12))
# test_pred = np.zeros((X_test.shape[0],12))
# for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
#     print(index)
#     train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
#     y_val = cv_model[index].predict_proba(test_x[feature_name])
#     print(y_val.shape)
#     cv_pred[test_index] = y_val
#     test_pred += cv_model[index].predict_proba(X_test[feature_name]) / 5
#
# print(np.mean(cv_score))
#
# oof_train = DF(cv_pred)
# # oof_train.columns = ['label_'+str(i) for i in range(0,12)]
# oof_train['sid'] = all_data[all_data['click_mode'].notnull()]['sid'].values
# oof_train[12] = y
# # oof_train['click_mode'] = all_data[tr_index]['click_mode'].reset_index(drop=True)
# oof_train.set_index('sid',inplace=True)
#
# oof_test = DF(test_pred)
# # oof_test.columns = ['label_'+str(i) for i in range(0,12)]
# oof_test['sid'] = all_data[~tr_index]['sid'].values
# oof_test[12] = np.nan
# oof_test.set_index('sid',inplace=True)
#
# oof_train.to_hdf("../cache_data/stacking_{}_fold_{}_feature_phase2.hdf".format(cv,len(feature_name)),'train')
# oof_test.to_hdf("../cache_data/stacking_{}_fold_{}_feature_phase2.hdf".format(cv,len(feature_name)),'test')
#
# # Offline 后处理前
#
# if version == 2:
#     train_clicks_2 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version),parse_dates=['click_time'],nrows=nrows)
#     train_clicks_1 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version-1),parse_dates=['click_time'],nrows=nrows)
#     answer = train_clicks_2.append(train_clicks_1).reset_index(drop=True).fillna(0)
# else:
#     answer = pd.read_csv(input_dir+'train_clicks.csv',parse_dates=['click_time'],nrows=nrows)
#
# if offline:
#     answer = all_data[~tr_index][['sid','city']].merge(answer,how='left',on='sid').fillna(0)
#     answer['pred'] = np.argmax(test_pred,axis=1)
#     print(f1_score(answer['click_mode'],answer['pred'],average='weighted'))
#
# for i in range(0,4):
#     tmp = answer[answer['city']==i]
#     print(i,f1_score(tmp['click_mode'],tmp['pred'],average='weighted'))
#
# num_classes = 12
# label_name = 'click_mode'
# oof_train.rename(columns={num_classes:label_name,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11},inplace=True)
# oof_test.rename(columns={num_classes:label_name,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11},inplace=True)
#
# tmp_train = oof_train[oof_train.index.isin(space_time[space_time['req_time']>='2018-11-15']['sid'].unique())]
# sz_train = tmp_train[tmp_train.index.isin(space_time[space_time['city']==2]['sid'].unique())]
# other_train = tmp_train.copy()#[tmp_train.index.isin(space_time[space_time['city']!=2]['sid'].unique())]
# raw_train_score = f1_score(tmp_train[label_name],np.argmax(tmp_train[range(num_classes)].values,axis=1),average='weighted')
# #raw_valid_score = f1_score(valid[label_name],np.argmax(valid[range(num_classes)].values,axis=1),average='weighted')
#
# print("RAW SCORE: ",raw_train_score)#raw_valid_score
#
# class OptimizedRounder(object):
#     def __init__(self):
#         self.coef_ = 0
#         self.coef_arr = []
#         self.val_score = []
#
#     def _kappa_loss(self, coef, X, y):
#         X_p = DF(np.copy(X))
#         for i in range(len(coef)):
#             X_p[i] *= coef[i]
#
#         l1 = f1_score(y, np.argmax(X_p.values,axis=1), average="weighted")
#         self.coef_arr.append(coef)
#
#         print(list(coef.astype(np.float16)),' Train score = ',l1.astype(np.float32))#,' Valid score =',l2.astype(np.float16))
#         return -l1
#
#     def fit(self, X, y):
#         loss_partial = partial(self._kappa_loss, X=X, y=y)
#         self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Powell')
#
#     def predict(self, X, coef):
#         X_p = DF(np.copy(X))
#         for i in range(len(coef)):
#             X_p[i] *= coef[i]
#         return X_p
#
#     def coefficients(self):
#         return self.coef_['x']
#
# # 下面是一起处理的
# # cv_pred = tmp_train[range(num_classes)].values
# # y = tmp_train[label_name].values
# # initial_coef = [1.0000] * num_classes
#
# # optR = OptimizedRounder()
# # optR.fit(cv_pred, y)
# # best_score = optR.coefficients()
#
# # best_coef = optR.coefficients()
# # print(best_coef)#,best_score
#
# # SZ
#
# cv_pred = sz_train[range(num_classes)].values
# y = sz_train[label_name].values
# initial_coef = [1.1] * num_classes
#
# optR = OptimizedRounder()
# optR.fit(cv_pred, y)
# sz_score = optR.coefficients()
#
# # Other
#
# cv_pred = other_train[range(num_classes)].values
# y = other_train[label_name].values
# initial_coef = [1.1] * num_classes
#
# optR = OptimizedRounder()
# optR.fit(cv_pred, y)
# other_score = optR.coefficients()
#
# sz_test = oof_test[oof_test.index.isin(space_time[space_time['city']==2]['sid'].unique())]
# other_test = oof_test[oof_test.index.isin(space_time[space_time['city']!=2]['sid'].unique())]
# print(sz_test.shape,other_test.shape)
#
# sz_y = list(sz_train[label_name].values)
# other_y = list(other_train[label_name].values)
# y = sz_y + other_y
#
# sz_train = optR.predict(sz_train[range(num_classes)].values,sz_score)
# other_train = optR.predict(other_train[range(num_classes)].values,other_score)
# cv_pred = sz_train.append(other_train)
#
# print("Global Best")
# print(best_coef)
# print("\nValid Counts = ", Counter(y))
# print("Predicted Counts = ", Counter(np.argmax(cv_pred.values,axis=1)))
# acc1 = raw_train_score
# acc2 = f1_score(y,np.argmax(cv_pred.values,axis=1),average="weighted")
# print("Train Before = ",acc1)
# print("Train After = ",acc2)
# print("Train GAP = ",acc2-acc1)
#
# test_pred = optR.predict(oof_test[range(num_classes)], best_coef)
# test_pred = np.argmax(test_pred.values,axis=1)
#
# # 后处理后
#
# if version == 2:
#     train_clicks_2 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version),parse_dates=['click_time'],nrows=nrows)
#     train_clicks_1 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version-1),parse_dates=['click_time'],nrows=nrows)
#     answer = train_clicks_2.append(train_clicks_1).reset_index(drop=True).fillna(0)
# else:
#     answer = pd.read_csv(input_dir+'train_clicks.csv',parse_dates=['click_time'],nrows=nrows)
#
# if offline:
#     answer = all_data[~tr_index][['sid','city']].merge(answer,how='left',on='sid').fillna(0)
#     answer['pred'] = test_pred
#     print(f1_score(answer['click_mode'],answer['pred'],average='weighted'))
#
# for i in range(0,4):
#     tmp = answer[answer['city']==i]
#     print(i,f1_score(tmp['click_mode'],tmp['pred'],average='weighted'))
#
# ALL