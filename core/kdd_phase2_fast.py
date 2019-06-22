#!/apps/dslab/anaconda/python3/bin/python
# -*- coding: utf-8 -*-
# 推荐场景
# 	针对 PID / O D / TIME 字段 推荐用户出行模式
# 		1. 百度内置模型
# 		2. 基于百度内置模型衍生特征模型
# 		3. 与Plans无关的模型
#       4. OD模型与百度内置模型的交叉

# Stacking需要学习的是 不同维度刻画下的，对当前Query(PID,O,D,TIME)的推荐
# No Graph Embedding Version

import json
import time
import warnings
from collections import Counter
from math import radians, atan, tan, sin, acos, cos, atan2, sqrt

import geohash
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame as DF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

#%matplotlib inline

# GLOBAL Param
cv = 5               # CV Folds, used in ratio and model train
random_seed = 2019   # Random Seed
for_test = True      # Control Test-Set
offline = True       # Use 11.24 - 11.31 For Offline Test
version = 2          # Phase

if for_test:
    nrows = None
else:
    nrows = 10000

input_dir = '../input/data_set_phase{}/'.format(version)

t1 = time.time()
print("Now Input Data...")

if version == 2:
    profiles = pd.read_csv(input_dir+'profiles.csv',nrows=nrows)
    train_clicks_2 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version),parse_dates=['click_time'],nrows=nrows)
    train_clicks_1 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version-1),parse_dates=['click_time'],nrows=nrows)
    train_clicks = train_clicks_2.append(train_clicks_1).reset_index(drop=True)
    train_plans_2 = pd.read_csv(input_dir+'train_plans_phase{}.csv'.format(version),parse_dates=['plan_time'],nrows=nrows)
    train_plans_1 = pd.read_csv(input_dir+'train_plans_phase{}.csv'.format(version-1),parse_dates=['plan_time'],nrows=nrows)
    train_plans = train_plans_2.append(train_plans_1).reset_index(drop=True)
    train_queries_2 = pd.read_csv(input_dir+'train_queries_phase{}.csv'.format(version),parse_dates=['req_time'],nrows=nrows)
    train_queries_1 = pd.read_csv(input_dir+'train_queries_phase{}.csv'.format(version-1),parse_dates=['req_time'],nrows=nrows)
    train_queries = train_queries_2.append(train_queries_1).reset_index(drop=True)
    test_plans = pd.read_csv(input_dir+'test_plans.csv',parse_dates=['plan_time'],nrows=nrows)
    test_queries = pd.read_csv(input_dir+'test_queries.csv',parse_dates=['req_time'],nrows=nrows)
    print("Use Time {}".format(time.time()-t1))
else:
    profiles = pd.read_csv('../input/data_set_phase2/profiles.csv',nrows=nrows)
    train_clicks = pd.read_csv('../input/data_set_phase2/train_clicks_phase1.csv',parse_dates=['click_time'],nrows=nrows)
    train_plans = pd.read_csv('../input/data_set_phase2/train_plans_phase1.csv',parse_dates=['plan_time'],nrows=nrows)
    train_queries = pd.read_csv('../input/data_set_phase2/train_queries_phase1.csv',parse_dates=['req_time'],nrows=nrows)
    test_plans = pd.read_csv('../input/data_set_phase2/test_plans.csv',parse_dates=['plan_time'],nrows=nrows)
    test_queries = pd.read_csv('../input/data_set_phase2/test_queries.csv',parse_dates=['req_time'],nrows=nrows)
    print("Use Time {}".format(time.time()-t1))

print(profiles.shape,train_clicks.shape,train_plans.shape,train_queries.shape)
print(test_plans.shape,test_queries.shape)

if offline:
    tmp = train_queries[train_queries['req_time']<'2018-12-01']
    train_queries = tmp[tmp['req_time']<'2018-11-24']
    test_queries = tmp[tmp['req_time']>='2018-11-24']
    del tmp;

def jsonLoads(strs,key):
    '''strs：传进来的json数据
       key：字典的键
    '''
    try:
        ret = []
        dict_ = json.loads(strs)
        for i in dict_:
            if i[key]=='':
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
    
def flatten_data(col):
    """
    把plans  flatten
    """
    df = pd.DataFrame(list(plans[col].values))
    df['sid'] = plans['sid']
    dis = pd.DataFrame()
    for i in df.columns[:-1]:
        dis_df = df.loc[:,[i,'sid']].copy()
        dis_df.columns = [col,'sid']
        dis = pd.concat([dis,dis_df],axis=0  )
    dis = dis.dropna()
#     dis = dis.sort_values('sid').reset_index(drop = True)
    return dis

print("Deal With Plans...")
plans = pd.concat([train_plans,test_plans],axis=0).reset_index(drop = True)
queries = pd.concat([train_queries,test_queries],axis=0).reset_index(drop = True)
train_clicks = train_clicks.merge(train_queries[['sid']],how='right',on='sid')
train_clicks.fillna(0,inplace=True)
print(train_clicks.shape)

data = train_clicks[['sid','click_mode']].copy()
test_id = test_queries[['sid']].copy()
data = pd.concat([data,test_id],axis=0).fillna(-1).reset_index(drop = True)
plans = data[['sid']].merge(plans,on='sid',how='left').reset_index(drop = True)
queries = data[['sid']].merge(queries,on='sid',how='left').reset_index(drop = True)

for i in tqdm(['distance','price','eta','transport_mode']):
    plans[i] = plans['plans'].apply(jsonLoads, key=i)

distance = flatten_data(col = 'distance')
price = flatten_data(col = 'price')
price.replace('',np.nan,inplace=True)
eta = flatten_data(col = 'eta')
transport_mode = flatten_data(col = 'transport_mode')

"""transport_mode_rank"""
plans['transport_mode_rank'] = plans['transport_mode'].apply(lambda x:np.arange(len(x)))
plans['distance_rank'] = plans['distance'].apply(lambda x:np.argsort(x))
plans['price_rank'] = plans['price'].apply(lambda x:np.argsort(x))
plans['eta_rank'] = plans['eta'].apply(lambda x:np.argsort(x))
plans['transport_mode_str'] = plans['transport_mode'].astype('str')
plans['price_str'] = plans['price'].astype('str')
plans['distance_str'] = plans['distance'].astype('str')
plans['eta_str'] = plans['eta'].astype('str')

transport_mode_rank = flatten_data(col = 'transport_mode_rank')

plans_df = pd.concat([distance,transport_mode_rank.iloc[:,0],eta.iloc[:,0],transport_mode.iloc[:,0],price.iloc[:,0]],axis=1)

transport_mode_list = plans[['sid','transport_mode']].copy()
transport_mode_list.columns = ['sid','transport_mode_list']
plans_df = plans_df.merge(plans[['sid','plan_time']], on='sid',how='left')
print(plans_df.shape)

data = data.merge(plans_df, on='sid',how='left')
data = data.merge(queries, on='sid',how='left')
data['ep'] = data['eta'] / data['price'] # 单位时间所需价格
data['dp'] = data['distance'] / data['price'] # 单位距离所需价格
data['de'] = data['distance'] / data['eta'] # 单位距离所需时间
data['ed'] = data['eta'] / data['distance'] # 单位eta所需距离
data['pe'] = data['price'] / data['eta'] 
data['pd'] = data['price'] / data['distance']
print(data.shape,data.columns)
print("Plans Prepare Finished...")

# OD 

def getDistance(latA, lonA, latB, lonB):  # 球面距离
    ra = 6378140     # radius of equator: meter  
    rb = 6356755     # radius of polar: meter  
    flatten = (ra - rb) / ra   # Partial rate of the earth  
    # change angle to radians  
    radLatA = radians(latA)  
    radLonA = radians(lonA)  
    radLatB = radians(latB)  
    radLonB = radians(lonB)  
  
    try: 
        pA = atan(rb / ra * tan(radLatA))  
        pB = atan(rb / ra * tan(radLatB))  
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))  
        c1 = (sin(x) - x) * (sin(pA) + sin(pB))**2 / cos(x / 2)**2  
        c2 = (sin(x) + x) * (sin(pA) - sin(pB))**2 / sin(x / 2)**2  
        dr = flatten / 8 * (c1 - c2)  
        distance = ra * (x + dr)  
        return distance   # meter   
    except:
        return np.nan

def bearing(lat1, lng1, lat2, lng2): # 角度特征
    AVG_EARTH_RADIUS = 6378.137  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def get_city(local): # 城市编码
    local = local.split(',')
    x = float(local[0])
    y = float(local[1])
    d1 = (x-116.41) ** 2 + (y-39.91) ** 2
    d2 = (x-121.43) ** 2 + (y-31.20) ** 2
    d3 = (x-114.06) ** 2 + (y-22.54) ** 2
    d4 = (x-113.26) ** 2 + (y-23.13) ** 2
    distance = [d1,d2,d3,d4]
    return np.argmin(distance)

def cal_manhattan_distance(O_lon, O_lat, D_lon, D_lat): # 曼哈顿距离
    dlat = O_lat - D_lat
    a = sin(dlat / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371
    lat_d = c * r

    dlon = O_lat - D_lat
    a = sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371
    lon_d = c * r

    return lat_d + lon_d

base32 = {x:i+1 for i,x in enumerate(list('0123456789bcdefghjkmnpqrstuvwxyz') )} # GEOHASH 配置文件
def geohash2int(geohash_id):
    result = 0
    base = 1
    for each in geohash_id[::-1]:
        result = result + base32[each] * base
        base = base*32
    return result

def get_ktime_feature(k,data,i): # 排名为I时的特征
    kfc = data.copy()
    for time in range(1,k):
        tmp = kfc.sort_values(by=['sid',i]).drop_duplicates(subset=['sid'],keep='first')
        kfc = kfc[~kfc.index.isin(tmp.index)]
    tmp = kfc.sort_values(by=['sid',i]).drop_duplicates(subset=['sid'],keep='first')
    return tmp

from scipy import stats, dot, linalg


def get_mode(x):  # 众数
    return stats.mode(x)[0][0]

def get_mode_count(x):  # 众数的统计值
    return stats.mode(x)[1][0]
#
# # Graph Embedding
# def get_graph_embedding(data=None,cols=None,emb_size=128,isWeight=False,model_type=None,weight_col=[],isGraph=False,intGraph=None):
#
#     for i in tqdm([i for i in cols if i not in weight_col]):
#         data[i] = data[i].astype('str')
#     for i in weight_col:
#         data[i] = data[i].astype('int')
#     if isGraph:
#         G = intGraph
#     else:
#         G = nx.DiGraph()
#         if isWeight:
#             G.add_weighted_edges_from(data[cols].drop_duplicates(subset=cols,keep='first').values)
#         else:
#             G.add_edges_from(data[cols].drop_duplicates(subset=cols,keep='first').values)
#     if model_type == 'Node2Vec':
#         model = Node2Vec(G, walk_length = 10, num_walks = 100,p = 0.25, q = 4, workers = 1)#init model
#         model.train(window_size = 5, iter = 3)# train model
#     elif model_type == 'DeepWalk' :
#         model = DeepWalk(G,walk_length=10,num_walks=80,workers = 1)#init model
#         model.train(window_size=5,iter=3)# train model
#     elif model_type == "SDNE" :
#         model = SDNE(G,hidden_size=[256,128]) #init model
#         model.train(batch_size=3000,epochs=40,verbose=2)# train model
#     elif model_type == "LINE":
#         model = LINE(G,embedding_size=128,order='second') #init model,order can be ['first','second','all']
#         model.train(batch_size=1024,epochs=50,verbose=2)# train model
#     elif model_type == "Struc2Vec" :
#         model = Struc2Vec(G, 10, 100, workers=1, verbose=40, ) #init model
#         model.train(window_size = 5, iter = 3)# train model
#
#     embeddings = model.get_embeddings()# get embedding vectors
# #     evaluate_embeddings(embeddings)
#     embeddings = pd.DataFrame(embeddings).T
#     new_col = "".join(cols)
#     embeddings.columns = ['{}_{}_emb_{}'.format(new_col,model_type,i) for i in embeddings.columns]
#     embeddings = embeddings.reset_index().rename(columns={'index' : 'node_{}'.format(cols[0])})
#
#     return embeddings

#####  特征工程部分 #####

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

# Plans Feature Distance
def get_stat(x):
    res = np.array([i for i in x if i!=0])
    if len(res)==0:
        return 0
    else:
        return res

def edit_distance(word1, word2):
    try:
        len1 = len(word1);
        len2 = len(word2);
        dp = np.zeros((len1 + 1,len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i;     
        for j in range(len2 + 1):
            dp[0][j] = j;

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i-1] == word2[j-1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))   
        return dp[len1][len2]
    except:
        return np.nan
    
def other_distance(word1, word2, param='jaccard'):
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
        
# def ham_distance(word1, word2):
#     try:
#         ham = np.nonzero(matV[0]-matV[1])
#         return np.shape(smstr[0])[1]
#     except:
#         return np.nan

def cos_distance(word1, word2):
    try:
        cosV12 = dot(word1,word2)/(linalg.norm(word1)*linalg.norm(word2))
        return cosV12
    except:
        return np.nan

def l2_distance(word1, word2):
    try:
        return sqrt((word1-word2)*((word1-word2).T))
    except:
        return np.nan

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
plans_feature['price_nonan_skew'] = plans['price'].map(lambda x:stats.skew(get_stat(x)))
plans_feature['price_nonan_kurt'] = plans['price'].map(lambda x:stats.kurtosis(get_stat(x)))
plans_feature['price_nonan_std'] = plans['price'].map(lambda x:np.std(get_stat(x)))
plans_feature['price_nonan_max'] = plans['price'].map(lambda x:np.max(get_stat(x)))
plans_feature['price_nonan_min'] = plans['price'].map(lambda x:np.min(get_stat(x)))
plans_feature['price_nonan_sum'] = plans['price'].map(lambda x:np.sum(get_stat(x)))

padding_maxlen = np.max(plans_feature['transport_mode_len'])

plans_feature['price_have_0_num'] = plans['price'].map(lambda x:len([i for i in x if i==0]))
plans_feature['price_have_0_ratio'] = plans_feature['price_have_0_num'] / plans_feature['transport_mode_len']
plans_feature['price_mean'] = plans['price'].map(lambda x:np.mean(x))
plans_feature['distance_mean'] = plans['distance'].map(lambda x:np.mean(x))
plans_feature['eta_mean'] = plans['eta'].map(lambda x:np.mean(x))

plans_feature['distance_min'] = plans['distance'].map(lambda x:np.min(x))
plans_feature['distance_max'] = plans['distance'].map(lambda x:np.max(x))
plans_feature['distance_std'] = plans['distance'].map(lambda x:np.std(x))
plans_feature['distance_sum'] = plans['distance'].map(lambda x:np.sum(x))
plans_feature['distance_skew'] = plans['distance'].map(lambda x:stats.skew(x))
plans_feature['distance_kurt'] = plans['distance'].map(lambda x:stats.kurtosis(x))

plans_feature['eta_min'] = plans['eta'].map(lambda x:np.min(x))
plans_feature['eta_max'] = plans['eta'].map(lambda x:np.max(x))
plans_feature['eta_std'] = plans['eta'].map(lambda x:np.std(x))
plans_feature['eta_sum'] = plans['eta'].map(lambda x:np.sum(x))
plans_feature['eta_skew'] = plans['eta'].map(lambda x:stats.skew(x))
plans_feature['eta_kurt'] = plans['eta'].map(lambda x:stats.kurtosis(x))

plans_feature['transport_mode_mode'] = plans['transport_mode'].map(lambda x:stats.mode(x))
plans_feature['transport_mode_mode_count'] = plans_feature['transport_mode_mode'].map(lambda x:x[1][0])
plans_feature['transport_mode_mode'] = plans_feature['transport_mode_mode'].map(lambda x:x[0][0])
plans_feature['transport_mode_transform_count'] = plans_feature.groupby(['transport_mode_mode','transport_mode_mode_count'])['sid'].transform('count')

def get_padding(x,delta=0):
    if delta!=0:
        return list((x+delta)) + ([0]*(padding_maxlen-len(x)))
    else:
        return list((x)) + ([0]*(padding_maxlen-len(x)))
plans['distance_rank_array'] = plans['distance_rank'].map(lambda x:get_padding(x,1))
plans['eta_rank_array'] = plans['eta_rank'].map(lambda x:get_padding(x,1))
plans['price_rank_array'] = plans['price_rank'].map(lambda x:get_padding(x,1))
plans['mode_rank_array'] = plans['transport_mode_rank'].map(lambda x:get_padding(x,1))

plans['distance_array'] = plans['distance'].map(lambda x:get_padding(x,0))
plans['eta_array'] = plans['eta'].map(lambda x:get_padding(x,0))
plans['price_array'] = plans['price'].map(lambda x:get_padding(x,0))
plans['transport_mode_array'] = plans['transport_mode'].map(lambda x:get_padding(x,0))

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
    plans_feature['{}_l2_distance'.format('_'.join(i))] = list(map(lambda x,y:l2_distance(x,y),plans[i[0]],plans[i[1]]))
    plans_feature['{}_cos_distance'.format('_'.join(i))] = list(map(lambda x,y:cos_distance(x,y),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_ham_distance'.format('_'.join(i))] = list(map(lambda x,y:ham_distance(x,y),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_edit_distance'.format('_'.join(i))] = list(map(lambda x,y:edit_distance(x,y),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_jaccard_distance'.format('_'.join(i))] = list(map(lambda x,y:other_distance(x,y,'jaccard'),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_corr_distance'.format('_'.join(i))] = list(map(lambda x,y:other_distance(x,y,'correlation'),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_braycurtis_distance'.format('_'.join(i))] = list(map(lambda x,y:other_distance(x,y,'braycurtis'),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_match_distance'.format('_'.join(i))] = list(map(lambda x,y:other_distance(x,y,'matching'),plans[i[0]],plans[i[1]]))
#     plans_feature['{}_wminkowski_distance'.format('_'.join(i))] = list(map(lambda x,y:other_distance(x,y,'wminkowski'),plans[i[0]],plans[i[1]]))
    
#     'braycurtis', 'canberra', 'chebyshev', 'cityblock',
#     'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
#     'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
#     'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
#     'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

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
        col0.append(other_distance(i[0][j],i[1][j],'cosine'))
        col1.append(other_distance(i[0][j],i[1][j],'euclidean'))
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

N_COM = 5

def tokenize(data):
    tokenized_docs = [word_tokenize(doc) for doc in data]
    alpha_tokens = [[t.lower() for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [[lemmatizer.lemmatize(alpha) for alpha in doc] for doc in alpha_tokens]
    X_stem_as_string = [" ".join(x_t) for x_t in lem_tokens]
    return X_stem_as_string

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

print(text_feature.shape)

pid_stats = feature[['sid']].merge(queries[['sid','pid']],how='left',on='sid')
pid_stats = pid_stats.merge(profiles,on='pid',how='left')
print(pid_stats.shape)

tmp = data[['pid','transport_mode']].groupby(['pid'])['transport_mode'].agg(['median','std','nunique','count',get_mode,get_mode_count]).add_prefix('pid_transport_mode_').reset_index()
pid_stats = pid_stats.merge(tmp,how='left',on='pid')

N_COM = 10
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

print(pid_stats.shape)

label = train_clicks[['click_mode','sid']].sort_values(by=['sid']).reset_index(drop=True)
def not_sid_col(x):
    return x[[i for i in x.columns if i not in ['sid','click_mode']]]

all_data = pd.concat([not_sid_col(pid_stats),not_sid_col(feature),
                      not_sid_col(plans_feature),not_sid_col(text_feature),
                      not_sid_col(space_time),#not_sid_col(fea),
                      label],axis=1)
print(all_data.shape)

feature_name = [i for i in all_data.columns if i not in ['sid','click_mode','plan_time','req_time']]
cate_feature = ['oy','ox','dx','dy','pid','p0','o','d','o_geohash','d_geohash','req_time_dow','req_is_weekend','sphere_dis_bins']

for i in tqdm(cate_feature):
    try:
        lbl = LabelEncoder()
        all_data[i] = lbl.fit_transform(all_data[i].astype('str'))
    except:
        print(i)
        continue
    
print(len(cate_feature),' ',len(feature_name))

#### Mode Train ####

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

# CV TRAIN

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
    print(index)
    lgb_model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
        max_depth=-1, n_estimators=3000, objective='multiclass',num_classes=12,
        subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
        learning_rate=0.05, random_state=2019 + index, n_jobs=40, metric="None", importance_type='gain'
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

import matplotlib.pyplot as plt

fi = []
for i in cv_model:
    tmp = {
        'name' : feature_name,
        'score' : i.feature_importances_
    }
    fi.append(pd.DataFrame(tmp))
    
fi = pd.concat(fi)
fig = plt.figure(figsize=(7,5))
# fi.groupby(['name'])['score'].agg('sum').sort_values(ascending=False).head(20).plot.barh()
fi.groupby(['name'])['score'].agg('mean').sort_values(ascending=False).head(20).plot.barh()

# Get CV
cv_pred = np.zeros((X_train.shape[0],12))
test_pred = np.zeros((X_test.shape[0],12))
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    print(index)
    train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    y_val = cv_model[index].predict_proba(test_x[feature_name])
    print(y_val.shape)
    cv_pred[test_index] = y_val
    test_pred += cv_model[index].predict_proba(X_test[feature_name]) / 5

print(np.mean(cv_score))

oof_train = DF(cv_pred)
# oof_train.columns = ['label_'+str(i) for i in range(0,12)]
oof_train['sid'] = all_data[all_data['click_mode'].notnull()]['sid'].values
oof_train[12] = y
# oof_train['click_mode'] = all_data[tr_index]['click_mode'].reset_index(drop=True)
oof_train.set_index('sid',inplace=True)

oof_test = DF(test_pred)
# oof_test.columns = ['label_'+str(i) for i in range(0,12)]
oof_test['sid'] = all_data[~tr_index]['sid'].values
oof_test[12] = np.nan
oof_test.set_index('sid',inplace=True)





print("End Time {}".format(time.time()-t1))