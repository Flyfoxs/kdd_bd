# 推荐场景 
# 	针对 PID / O D / TIME 字段 推荐用户出行模式
# 		1. 百度内置模型
# 		2. 基于百度内置模型衍生特征模型
# 		3. 与Plans无关的模型
#       4. OD模型与百度内置模型的交叉

# Stacking需要学习的是 不同维度刻画下的，对当前Query(PID,O,D,TIME)的推荐

import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
import json
from sklearn.metrics import f1_score
import geohash
import time
import gc
import math
from tqdm import tqdm
from math import radians, atan, tan, sin, acos, cos, atan2, sqrt
from scipy import stats
from ge import DeepWalk,Struc2Vec,SDNE,LINE,Node2Vec
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# GLOBAL Param
cv = 5				 # CV Folds, used in ratio and model train
random_seed = 2019   # Random Seed
for_test = True      # Control Test-Set
offline = False   	 # Use 11.24 - 11.31 For Offline Test
version = 2       	 # Phase

if for_test:
    nrows = None
else:
    nrows = 100000

input_dir = '../input/data_set_phase{}/'.format(version)

import time
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
    profiles = pd.read_csv(input_dir+'profiles.csv',nrows=nrows)
    train_clicks = pd.read_csv(input_dir+'train_clicks.csv',parse_dates=['click_time'],nrows=nrows)
    train_plans = pd.read_csv(input_dir+'train_plans.csv',parse_dates=['plan_time'],nrows=nrows)
    train_queries = pd.read_csv(input_dir+'train_queries.csv',parse_dates=['req_time'],nrows=nrows)
    test_plans = pd.read_csv(input_dir+'test_plans.csv',parse_dates=['plan_time'],nrows=nrows)
    test_queries = pd.read_csv(input_dir+'test_queries.csv',parse_dates=['req_time'],nrows=nrows)
    print("Use Time {}".format(time.time()-t1))
    

print(profiles.shape,train_clicks.shape,train_plans.shape,train_queries.shape)
print(test_plans.shape,test_queries.shape)

if use_offline:
    tmp = train_queries[train_queries['req_time']<'2018-12-01']
    train_queries = tmp[tmp['req_time']<'2018-11-24']
    test_queries = tmp[tmp['req_time']>='2018-11-24']
    del tmp;

# For Plans

def jsonLoads(strs,key):
    '''strs：传进来的json数据
       key：字典的键
    '''
    try:
        dict_ = json.loads(strs)
        return list(i[key] for i in dict_)
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
        dis = pd.concat([dis,dis_df],axis=0,sort=False)
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
data = pd.concat([data,test_id],axis=0,sort=False).fillna(-1).reset_index(drop = True)
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

from scipy import stats

def get_mode(x):  # 众数
    return stats.mode(x)[0][0]

def get_mode_count(x):  # 众数的统计值
    return stats.mode(x)[1][0]

# Graph Embedding 
def get_graph_embedding(data=None,cols=None,emb_size=128,isWeight=False,model_type=None,weight_col=[],isGraph=False,intGraph=None):
    
    for i in tqdm([i for i in cols if i not in weight_col]):
        data[i] = data[i].astype('str')
    for i in weight_col:
        data[i] = data[i].astype('int')
    if isGraph:
        G = intGraph
    else:
        G = nx.DiGraph()
        if isWeight:
            G.add_weighted_edges_from(data[cols].drop_duplicates(subset=cols,keep='first').values)
        else:
            G.add_edges_from(data[cols].drop_duplicates(subset=cols,keep='first').values)
    if model_type == 'Node2Vec':
        model = Node2Vec(G, walk_length = 10, num_walks = 100,p = 0.25, q = 4, workers = 1)#init model
        model.train(window_size = 5, iter = 3)# train model
    elif model_type == 'DeepWalk' :
        model = DeepWalk(G,walk_length=10,num_walks=80,workers = 1)#init model
        model.train(window_size=5,iter=3)# train model
    elif model_type == "SDNE" :
        model = SDNE(G,hidden_size=[256,128]) #init model
        model.train(batch_size=3000,epochs=40,verbose=2)# train model
    elif model_type == "LINE":
        model = LINE(G,embedding_size=128,order='second') #init model,order can be ['first','second','all']
        model.train(batch_size=1024,epochs=50,verbose=2)# train model
    elif model_type == "Struc2Vec" :
        model = Struc2Vec(G, 10, 100, workers=1, verbose=40, ) #init model
        model.train(window_size = 5, iter = 3)# train model
        
    embeddings = model.get_embeddings()# get embedding vectors
#     evaluate_embeddings(embeddings)
    embeddings = pd.DataFrame(embeddings).T
    new_col = "".join(cols)
    embeddings.columns = ['{}_{}_emb_{}'.format(new_col,model_type,i) for i in embeddings.columns]
    embeddings = embeddings.reset_index().rename(columns={'index' : 'node_{}'.format(cols[0])})

    return embeddings

# Word Embedding 


# TFIDF Embedding


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

# 百度推荐模型的向量特征
# 第一个Price的NaN出现的位置,共有几个价格的NaN,占长度的多少
# Price/Distance/Mean向量的 Mean/Std/Min/Max/Sum/Skew
# 针对序列如价格 [100,300,200,400,500] ---> [0,2,1,3,4] + 1 And Padding ---> [1,3,2,4,5,0,0] Binary-Encoder 

plans_feature = plans[['sid']]
plans_feature['mode_array_count_sid'] = plans.groupby(['transport_mode_str'])['sid'].transform('count')
plans_feature['price_count_sid'] = plans.groupby(['price_str'])['sid'].transform('count')
plans_feature['eta_count_sid'] = plans.groupby(['eta_str'])['sid'].transform('count')
plans_feature['distance_count_sid'] = plans.groupby(['distance_str'])['sid'].transform('count')
plans_feature['mode_price_count'] = plans.groupby(['transport_mode_str','price_str'])['sid'].transform('count')
plans_feature['mode_eta_count'] = plans.groupby(['transport_mode_str','eta_str'])['sid'].transform('count')
plans_feature['mode_distance_count'] = plans.groupby(['transport_mode_str','distance_str'])['sid'].transform('count')

def get_stat(x):
    res = [i for i in x if i!=0]
    if len(res)==0:
        return 0
    else:
        return res

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

# 5863种 transport_mode_array
def get_padding(x,delta=0):
	return list((x+delta)) + ([0]*(padding_maxlen-len(x)))
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
plans_feature = plans_feature.sort_values(by=['sid'])

# 时空
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

# Graph Embedding Feature

# 同构图
# No Weight
to_emb = ['o','d','req_time_hour_0','req_time_dow','ox','oy','dx','dy','pid']
space_time_odh = space_time[to_emb+['sid']]
space_time_odh['od'] = space_time_odh['o'] + ',' + space_time_odh['d']

embedding_od_n2v = get_graph_embedding(data = space_time_odh, cols=['o','d'], emb_size=8, isWeight=False, model_type='Node2Vec')
embedding_oxdx_n2v = get_graph_embedding(data = space_time_odh, cols=['ox','dx'], emb_size=8, isWeight=False, model_type='Node2Vec')
embedding_oydy_n2v = get_graph_embedding(data = space_time_odh, cols=['oy','dy'], emb_size=8, isWeight=False, model_type='Node2Vec')

space_time_odh['o_hour'] = space_time_odh['o'] + "_" + space_time_odh['req_time_hour_0'].astype('str')
space_time_odh['d_hour'] = space_time_odh['d'] + "_" + space_time_odh['req_time_hour_0'].astype('str')

space_time_odh['o_dow'] = space_time_odh['o'] + "_" + space_time_odh['req_time_dow'].astype('str')
space_time_odh['d_dow'] = space_time_odh['d'] + "_" + space_time_odh['req_time_dow'].astype('str')

space_time_odh['o_pid'] = space_time_odh['o'] + "_" + space_time_odh['pid'].astype('str')
space_time_odh['d_pid'] = space_time_odh['d'] + "_" + space_time_odh['pid'].astype('str')

embedding_od_hour_n2v = get_graph_embedding(data = space_time_odh, cols=['o_hour','d_hour'], emb_size=8, isWeight=False, model_type='Node2Vec')
embedding_od_dow_n2v = get_graph_embedding(data = space_time_odh, cols=['o_dow','d_dow'], emb_size=8, isWeight=False, model_type='Node2Vec')
embedding_od_pid_n2v = get_graph_embedding(data = space_time_odh, cols=['o_pid','d_pid'], emb_size=8, isWeight=False, model_type='Node2Vec')

# Weight
space_time_odh['weight'] = space_time_odh.groupby(['o'])['d'].transform('count')
embedding_od_n2v_weight = get_graph_embedding(data = space_time_odh, cols=['o','d','weight'], emb_size=8, isWeight=True, weight_col=['weight'],model_type='Node2Vec')

space_time_odh['weight'] = space_time_odh.groupby(['o_hour'])['d_hour'].transform('count')
embedding_od_hour_n2v_weight = get_graph_embedding(data = space_time_odh, cols=['o_hour','d_hour','weight'], emb_size=8, isWeight=True, weight_col=['weight'],model_type='Node2Vec')

space_time_odh['weight'] = space_time_odh.groupby(['o_dow'])['d_dow'].transform('count')
embedding_od_dow_n2v_weight = get_graph_embedding(data = space_time_odh, cols=['o_dow','d_dow','weight'], emb_size=8, isWeight=True, weight_col=['weight'],model_type='Node2Vec')

# 二分图 考虑Struc2Vec
embedding_od_s2v = get_graph_embedding(data = space_time_odh, cols=['o','d'], emb_size=4, isWeight=False, model_type='Struc2Vec')
embedding_od_pid_s2v = get_graph_embedding(data = space_time_odh, cols=['od','pid'], emb_size=4, isWeight=False, model_type='Struc2Vec')
embedding_hour_pid_s2v = get_graph_embedding(data = space_time_odh, cols=['pid','req_time_hour_0'], emb_size=4, isWeight=False, model_type='Struc2Vec')
embedding_dow_pid_s2v = get_graph_embedding(data = space_time_odh, cols=['pid','req_time_dow'], emb_size=4, isWeight=False, model_type='Struc2Vec')
embedding_opid_odow_s2v = get_graph_embedding(data = space_time_odh, cols=['opid','odow'], emb_size=4, isWeight=False, model_type='Struc2Vec')

# Merge Feature
space_time_odh = space_time_odh.merge(embedding_od_n2v,how='left',on=['o']).\
                                merge(embedding_oxdx_n2v,how='left',on=['ox']).\
                                merge(embedding_oydy_n2v,how='left',on=['oy']).\
                                merge(embedding_od_hour_n2v,how='left',on=['o_hour']).\
                                merge(embedding_od_dow_n2v,how='left',on=['o_dow']).\
                                merge(embedding_od_pid_n2v,how='left',on=['o_pid']).\
                                merge(embedding_od_n2v_weight,how='left',on=['o']).\
                                merge(embedding_od_hour_n2v_weight,how='left',on=['o_hour']).\
                                merge(embedding_od_dow_n2v_weight,how='left',on=['o_dow']).\
                                merge(embedding_od_s2v,how='left',on=['o']).\
                                merge(embedding_od_pid_s2v,how='left',on=['od']).\
                                merge(embedding_hour_pid_s2v,how='left',on=['pid']).\
                                merge(embedding_dow_pid_s2v,how='left',on=['pid']).\
                                merge(embedding_opid_odow_s2v,how='left',on=['opid'])

space_time_odh = space_time_odh[['sid'] + [i for i in space_time_odh.columns if 'emb' in i]]

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


#  OD PID Embedding (On Plans/Data)
 
#  OD Word2C Embdding (On Plans)

#  Data 与 PID / OD 的交叉统计

data 

#  判定类特征
    1. transport_mode_mode == Recommand_0
    2. price_max==0 && length>1 # 多个不用钱的手段
