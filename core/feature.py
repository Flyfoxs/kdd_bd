import sys
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

from core.config import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache
from functools import lru_cache

from glob import glob
import json
from math import radians, atan, tan, sin, acos, cos

import warnings
warnings.filterwarnings("ignore")

# #from core.submit import *
# from core.check import *
# from core.predict import *
# from core.merge import *
# from core.merge import *



# logging.getLogger().setLevel(logging.INFO)

# import argparse
# sys.argv = ['program',  '-W' , '--gp_name', 'lr_bin_9', '--shift', 0]

# args = check_options()

# print(args)
# args.dynamic

# logging.getLogger().setLevel(logging.INFO)


# Variable


from  pandas import Series



@file_cache()
def get_plan_model_sequence():
    deep_plan = get_plan_original_deep() #get_plan_model_sequence
    deep_plan = deep_plan.set_index('dummy_sid')
    res_list = []
    #res_list.append(deep_plan.dummy_sid)
    for i in range(12):
        if i == 0:
            sel_col = ['en_label', 'transport_mode', ]
        else:
            sel_col = ['transport_mode']
        recomm_mode = deep_plan.loc[deep_plan.sn==i, sel_col ]
        recomm_mode.rename({'transport_mode':f'o_seq_{i}'}, inplace = True, axis=1)
        if recomm_mode.iloc[:,-1].sum()==0 :
            logger.info(f'No sid have more than {i} plan')
            break
        res_list.append(recomm_mode)


    return pd.concat(res_list, axis=1)

@file_cache()
def get_plan_mini(model):
    """
    split the jasn to 11 model group
    :param plan_file:
    :param model:
    :return:
    """

    deep_plan= get_plan_original_deep() #get_plan_mini

    sigle_model = deep_plan.loc[deep_plan.transport_mode==model]#.copy()
    old_len = len(sigle_model)
    sigle_model = sigle_model.drop_duplicates('dummy_sid')

    logger.info(f'There are {old_len - len(sigle_model)} records were remove from {old_len} for mode:{model}')

    sigle_model['price'] = sigle_model['price'].fillna(0)  # .astype(float)

    sigle_model = sigle_model.fillna(0).astype(int)#.set_index('dummy_sid')

    mini_plan = sigle_model.loc[:, plan_items]

    mini_plan['pe_eta_price'] = mini_plan['eta']/mini_plan['price']
    mini_plan['pe_dis_price'] = mini_plan['distance']/mini_plan['price']
    mini_plan['pe_dis_eta']   = mini_plan['distance']/mini_plan['eta']

    mini_plan['pe_price_eta'] = mini_plan['price']/mini_plan['eta']
    mini_plan['pe_price_dis'] = mini_plan['price']/mini_plan['distance']
    mini_plan['pe_eta_dis']   = mini_plan['eta']/mini_plan['distance']

    #mini_plan['transport_mode'] = mini_plan['transport_mode']//model

    mini_plan['sid']       = sigle_model.sid
    mini_plan['dummy_sid'] = sigle_model.dummy_sid

    mini_plan = mini_plan.set_index(['dummy_sid', 'sid'])

    return mini_plan.add_prefix(f'{model}_')


@lru_cache()
def get_original(file):
    return pd.read_csv(f'{input_folder}/{file}', dtype=type_dict)


#
# def get_convert_profile_click_percent(feature):
#     col = sorted([item for item in feature.columns if item.endswith('transport_mode')])
#     df_col = col.copy()
#     df_col.extend(['click_mode', 'pid','day'])
#     # Click
#     logger.info(f'Only base on day from 0 to {val_cut_point} to cal click percentage')
#     click = feature.loc[:, df_col].copy()
#     for sn, cur_col in enumerate(tqdm(col, 'Click sum') ):
#         #print(cur_col)
#         click[cur_col] = click.apply(lambda item: 1 if item[cur_col] > 0 and item[cur_col] == item['click_mode'] else 0,
#                                      axis=1)
#
#     # Ignore
#     ignore = feature.loc[:, df_col].copy()
#     for sn, cur_col in  enumerate(tqdm(col, 'Ignore sum') ):
#         #print('ignore', cur_col)
#         ignore[cur_col] = ignore.apply(
#             lambda item: 1 if item[cur_col] > 0 and item[cur_col] != item['click_mode'] else 0, axis=1)
#
#     profile = pd.DataFrame()
#     for sn, cur_col in enumerate(tqdm(col, 'Cal percentage') ):
#         print(cur_col)
#         click_total = click.groupby(['pid','day'])[cur_col].agg({f'click_p_{sn+1:02}': 'sum'})
#         ignore_total = ignore.groupby(['pid','day'])[cur_col].agg({f'click_p_{sn+1:02}': 'sum'})
#         percent = click_total / (click_total + ignore_total)
#         # print(type(percent))
#         profile = pd.concat([profile, percent], axis=1)  # .fillna(0)
#     profile = profile.fillna(0).reset_index()
#     profile.day = profile.day+7
#     return profile


#
# def get_plan_summary():
#     plan = get_plan_original()
#     res_list = []
#     for item in ['distance']:
#         col_list = [col for col in plan.columns if col[1] == item]
#         summary = plan.loc[:, col_list].copy()
#         # max_ = summary.where(summary > 0).max(axis=1)
#         # max_.name=f'{item}_max'
#         # res_list.append(max_)
#         min_  = summary.where(summary > 0).min(axis=1)
#
#         min_ = pd.cut(min_,20).cat.codes
#         min_.name = f'{item}_min_cat'
#         res_list.append(min_)
#     res = pd.concat(res_list, axis=1)
#     res = res.sort_index(axis=1, level=1)
#     return res

#
# @timed()
# @file_cache()
# def get_plan_percentage_min():
#     """
#     Convert plan from amount/qty to percentage
#     :param plan:
#     :return:
#     """
#     plan = get_plan_original()
#     res_list = []
#     for item in plan_items_mini :
#         col_list = [col for col in plan.columns if col[1] == item]
#         plan_percent = plan.loc[:, col_list].copy()
#         total = plan_percent.where(plan_percent > 0).min(axis=1)
#         for col in plan_percent:
#             plan_percent[(str(col[0]), f'{col[1]}_min_p')] = round(plan_percent[col] / total, 4)
#             del plan_percent[col]
#
#         res_list.append(plan_percent)
#     res = pd.concat(res_list, axis=1)
#
#     # res.columns.set_levels([ f'{item[1]}_p' for item in res.columns ],level=1,inplace=True)
#     # res.columns = [ (item[0], f'{item[1]}_p') for item in res.columns]
#     res = res.sort_index(axis=1, level=1)
#     return res


@timed()
@file_cache()
def get_plan_percentage():
    """
    Convert plan from amount/qty to percentage
    :param plan:
    :return:
    """
    plan = get_plan_original_wide()
    res_list = []
    for item in plan_items_mini:
        #print(item)
        col_list = [col for col in plan.columns if col.endswith(item)]
        plan_percent = plan.loc[:, col_list].copy()
        #print(plan_percent.columns)
        total = plan_percent.max(axis=1)
        for col in plan_percent:
            plan_percent[f'{col}_max_p'] = round(plan_percent[col] / total, 4)
            del plan_percent[col]

        res_list.append(plan_percent)
    res = pd.concat(res_list, axis=1)
    return res


@file_cache()
def get_plan_nlp():

    N_COM = 10

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.pipeline import Pipeline
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    # def tokenize(data):
    #     tokenized_docs = [word_tokenize(doc) for doc in data]
    #     alpha_tokens = [[t.lower() for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    #     lemmatizer = WordNetLemmatizer()
    #     lem_tokens = [[lemmatizer.lemmatize(alpha) for alpha in doc] for doc in alpha_tokens]
    #     X_stem_as_string = [" ".join(x_t) for x_t in lem_tokens]
    #     return X_stem_as_string

    # vct = CountVectorizer(stop_words='english', lowercase=False)
    svd = TruncatedSVD(n_components=N_COM, random_state=2019)
    tfvec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=8000, analyzer='char',
                            lowercase=False)

    plans = get_plan_original_wide()
    plans.columns = [item[1] if isinstance(item, tuple) else item for item in plans.columns]
    plans = plans.reset_index()

    res_list = []
    for i in tqdm([#'distance', 'price', 'eta',
                 'plans', ]):
        x = plans[i].fillna(0)
        if i != 'plans':
            x = x.apply(lambda item: ','.join(item.astype(str)), axis=1).values
        print(x[:5])
        print(x.shape)
        # x = tokenize(x)
        print(len(x), x[:5])

        x = tfvec.fit_transform(x)
        x = svd.fit_transform(x)
        svd_feas = pd.DataFrame(x, index=plans.sid)
        svd_feas.columns = ['{}_svd_fea_{}'.format(i, j) for j in range(N_COM)]

        res_list.append(svd_feas)

        # feature = feature.merge(svd_feas, on='sid', how='left')
    res = pd.concat(res_list, axis=1)
    # print('====', res.columns)
    # print(res.shape)
    return res


@timed()
@file_cache()
def get_plans():
    """
    3, 5, 6 price is 0
    :return:
    """
    plan = get_plan_original_wide()

    plan_pg = get_plan_percentage()

    seq = get_plan_model_sequence()

    plan_stati = get_plan_stati_feature_sid()

    #todo plan_stati
    plan = pd.concat([plan, plan_pg, seq, plan_stati], axis=1)

    #plan.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plan.columns]

    return plan.reset_index()


@file_cache()
def get_plan_cat():
    plan_original = get_plan_original_wide()#.copy()

    last_col = plan_original.shape[1]
    for col in tqdm(plan_original.columns):
        #print(col)
        if col[1] != 'transport_mode' and  col != 'plans' and len(plan_original[col].value_counts()) > 100:
            plan_original['_'.join(col) + '_cat_code'] = pd.cut(plan_original[col], 100).cat.codes
        elif col[1] == 'transport_mode' or  col == 'plans':
            logger.info(('ignore', col))
        else:
            logger.info((f'\n{col}\n', plan_original[col].value_counts()))

    return plan_original.iloc[:,last_col:].reset_index()
#
# @file_cache()
# def get_plan_original_deep():
#     plan_list = []
#     for plan_file in ['train_plans.csv', 'test_plans.csv']:
#         original_plan = get_original(plan_file)
#         for sn, row in tqdm(original_plan.iterrows(), f'convert {plan_file}'):
#             plans = json.loads(row.plans)
#             plans_ex = entend_plan(plans, row.sid )
#             plan_list.extend(plans_ex)
#     res = pd.DataFrame(plan_list)
#     res = res.fillna('0').replace({'':'0'}).astype(int)#.set_index('dummy_sid')
#     res.index = res.dummy_sid
#
#
#     return res



@lru_cache()
def get_enhance_sid():
    plans = get_original('train_plans.csv')
    print(plans.shape)
    click = get_original('train_clicks.csv')

    plans = pd.merge(plans, click, how='left', on='sid')
    plans.click_mode = plans.click_mode.fillna(0).astype(int)
    plans = plans.loc[plans.click_mode.isin(enhance_model)]
    return dict(zip(plans.sid.values, plans.click_mode.values))

@file_cache()
def get_plan_original_deep():
    plan_list = []
    for plan_file in ['train_plans.csv', 'test_plans.csv']:
        original_plan = get_original(plan_file)
        normal_plan = original_plan.loc[~original_plan.sid.isin(get_enhance_sid())]
        for sn, row in tqdm(normal_plan.iterrows(), f'normal_plan {plan_file}, {len(normal_plan)}'):
            plans = json.loads(row.plans)
            plans_ex = entend_plan(plans, row.sid, False)
            plan_list.extend(plans_ex)

        extend_plan = original_plan.loc[original_plan.sid.isin(get_enhance_sid())]  # .head(2)
        for sn, row in tqdm(extend_plan.iterrows(), f'extend_plan {plan_file}, {len(extend_plan)}'):
            plans = json.loads(row.plans)
            plans_ex = entend_plan(plans, row.sid, True)
            plan_list.extend(plans_ex)
    res = pd.DataFrame(plan_list)
    res = res.fillna('0').replace({'': '0'}).astype(int)
    res.index = res.dummy_sid
    return res


dummy_sid = 1000


def entend_plan(plans, sid, enable_extend):
    sid = int(sid)

    global dummy_sid

    df_list = []
    plan_arr = [None] * 12
    full_mode_list = []
    for sn, single_plan in enumerate(plans):
        single_plan['sid'] = sid
        single_plan['sn'] = sn
        single_plan['en_label'] = 0
        single_plan['dummy_sid'] = dummy_sid
        df_list.append(single_plan.copy())
        transport_mode = int(single_plan['transport_mode'])
        # print(transport_mode, single_plan)
        plan_arr[transport_mode] = single_plan
        full_mode_list.append(transport_mode)

    dummy_sid += 1

    #
    # if enable_extend and sid in get_enhance_sid():
    #
    #     click_mode = get_enhance_sid()[sid]
    #
    #     # original_df = pd.DataFrame(plan_list)
    #
    #     remove_mode_list = full_mode_list.copy()
    #     if click_mode > 0:
    #         remove_mode_list.remove(click_mode)
    #     # 确保最后保留一个plan
    #     elif click_mode == 0 and len(remove_mode_list) == 1:
    #         remove_mode_list = []
    #
    #     for en_sn, remove_mode in enumerate(remove_mode_list):
    #         keep_mode = full_mode_list.copy()
    #         keep_mode.remove(remove_mode)
    #         # print(dummy_sid, keep_mode,  remove_mode, click_mode)
    #         for sn, model in enumerate(keep_mode):
    #             ex_plan = plan_arr.copy()[model]
    #             ex_plan['dummy_sid'] = dummy_sid
    #             ex_plan['en_label'] = en_sn + 1
    #             ex_plan['sn'] = sn
    #             df_list.append(ex_plan.copy())
    #         dummy_sid += 1

    #上采样
    if enable_extend and sid in get_enhance_sid():
        for en_sn in range(1, 4):
            for sn, single_plan in enumerate(plans):
                single_plan['sid'] = sid
                single_plan['sn'] = sn
                single_plan['en_label'] = en_sn
                single_plan['dummy_sid'] = dummy_sid
                df_list.append(single_plan.copy())
                transport_mode = int(single_plan['transport_mode'])
                # print(transport_mode, single_plan)
                plan_arr[transport_mode] = single_plan
                full_mode_list.append(transport_mode)

        dummy_sid += 1

    return df_list


def get_original_dummy_sid():
    deep_plan = get_plan_original_deep()
    return deep_plan.loc[deep_plan.en_label==0].index.values


@timed()
#@lru_cache()
@file_cache()
def get_plan_original_wide():
    res_list = []
    base_list = []
    for file in ['train_plans.csv', 'test_plans.csv']:
        base = get_original(file)
        base_list.append(base)
    base = pd.concat(base_list)
    from multiprocessing import Pool as ThreadPool  # 进程
    from functools import partial
    get_plan_mini_ex = partial(get_plan_mini)

    pool = ThreadPool(6)
    plan_list = pool.map(get_plan_mini_ex, tqdm(range(1, 12)), chunksize=1)


    plan_part = pd.concat(plan_list, axis=1)

    plan_part = plan_part.reset_index()

    plan_part['en_lable'] = 1

    plan_part.loc[plan_part.dummy_sid.isin(get_original_dummy_sid()) , 'en_lable'] = 0

    plan_part = pd.merge(base, plan_part, on='sid')
    return plan_part.set_index('dummy_sid')



@file_cache()
def get_query():
    import geohash as geo
    train = get_original('train_queries.csv')
    train['label'] = 'train'

    test = get_original('test_queries.csv')
    test['label'] = 'test'

    train_query = pd.concat([train, test])



    train_query.pid = train_query.pid.fillna(0)
    train_query.pid = train_query.pid.astype(int)
    train_query.req_time = pd.to_datetime(train_query.req_time)
    train_query['date'] = train_query.req_time.dt.date

    day_ = pd.to_datetime(train_query.req_time).dt.date
    day_ = day_ - min(day_)
    day_ = day_.dt.days

    train_query['day']  = day_
    train_query['weekday'] = train_query.req_time.dt.weekday
    train_query['hour'] = train_query.req_time.dt.hour
    train_query['weekend'] = train_query.weekday // 5

    train_query['o0'] = train_query.o.apply(lambda item: item.split(',')[0]).astype(float)
    train_query['o1'] = train_query.o.apply(lambda item: item.split(',')[1]).astype(float)

    train_query['d0'] = train_query.d.apply(lambda item: item.split(',')[0]).astype(float)
    train_query['d1'] = train_query.d.apply(lambda item: item.split(',')[1]).astype(float)

    train_query['sphere_dis'] = train_query.apply(lambda row: getDistance(row.o0,row.o1, row.d0,row.d1,), axis=1)

    for precision in [5,6]:
        train_query[f'o_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.o1, row.o0, precision=precision),
                                                               axis=1)

        train_query[f'd_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.d1, row.d0, precision=precision),
                                                               axis=1)

        train_query[f'o_d_hash_{precision}'] = train_query[f'o_hash_{precision}'] + '_' + train_query[f'd_hash_{precision}']

    return train_query

@timed()
def get_plan_stati_feature_sid():
    res_list = []

    #TODO, remove enhance plan
    plans = get_plan_original_deep() #get_plan_stati_feature_sid

    for col in plan_items_mini:
        tmp = plans.groupby('dummy_sid')[col].agg(['min', 'max', 'mean', 'std']).add_prefix(f'ps_{col}_')
        res_list.append(tmp)

    tmp = plans.groupby('dummy_sid')['transport_mode'].agg(['count','nunique']).add_prefix(f'ps_transport_mode_')
    tmp['ps_transport_mode_nunique'] = tmp['ps_transport_mode_count'] - tmp['ps_transport_mode_nunique']
    res_list.append(tmp)
    return pd.concat(res_list, axis=1)


@timed()
def get_stati_feature_pid():
    def get_mode(ser):
        return ser.value_counts().index[0]

    def get_mode_count(ser):
        return ser.value_counts().values[0]

    res_list = []

    query = get_query()
    query_mini = query.loc[:, ['pid', 'sid']]

    plans = get_plan_original_deep() #get_stati_feature_pid
    plans = pd.merge(plans, query_mini, how='left', on='sid')

    pid_mode = plans.groupby('pid').transport_mode.agg(
        ['median', 'std', 'nunique', 'count', get_mode, get_mode_count]).add_prefix('s_pid_m_')
    res_list.append(pid_mode)

    query = get_query()
    query.head()

    for direction in ['o', 'd']:
        query[f'{direction}_hash_6'] = query[f'{direction}_hash_6'].astype('category').cat.codes

        geohash_st = query.groupby('pid')[f'{direction}_hash_6'].agg(
            ['std', 'nunique', 'count', get_mode, get_mode_count]).add_prefix(f's_pid_{direction}_hash_')
        res_list.append(geohash_st)
    return pd.concat(res_list, axis=1).reset_index()

@timed()
def get_geo_percentage(query, direct, gp_level=[], prefix='glb_'):
    #hash_precision = precision

    print(gp_level)
    res_list = []
    for i in range(1, 12):
        tmp = query.groupby(gp_level)[f'{i}_transport_mode'].agg({f'sugg_{direct}_{i}': 'sum'})
        tmp[f'sugg_{direct}_{i}'] = tmp[f'sugg_{direct}_{i}'] // i

        res_list.append(tmp.astype(int))

    tmp = query.groupby(gp_level)['day'].agg(
        {f'day_appear_nunique_{direct}': 'nunique', f'count_appear_{direct}': 'count'})
    res_list.append(tmp.astype(int))

    tmp = pd.concat(res_list, axis=1)  # .sort_values('count_appear', ascending=False).loc['wx4g0w'].sort_index()

    for i in range(1, 12):
        tmp[f'sugg_{direct}_{i}_per'] = tmp[f'sugg_{direct}_{i}'] / tmp[f'count_appear_{direct}']

    tmp = tmp.add_prefix(prefix)
    # tmp.index.name = f'{direct}_hash_{hash_precision}'
    res = tmp.reset_index()

    return res


@lru_cache()
def get_click():
    return get_original('train_clicks.csv')

def get_profile():
    profile_data = get_original('profiles.csv').astype(int)
    # #profile_data = read_profile_data()
    # x = profile_data.drop(['pid'], axis=1).values
    # svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    # svd_x = svd.fit_transform(x)
    # svd_feas = pd.DataFrame(svd_x)
    # svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(20)]
    # svd_feas['pid'] = profile_data['pid'].values
    #
    # profile_data = pd.merge(profile_data,svd_feas, on='pid')

    lda = get_profile_lda()

    return pd.concat([profile_data,lda], axis=1)


@file_cache()
def get_profile_lda():
    def get_profile_text():
        profile = get_original('profiles.csv').astype(int)
        profile = profile.set_index('pid')
        col_list = profile.columns
        for col in tqdm(col_list):
            profile[col] = profile[col].map({0: ' ', 1: col})

        res = profile.apply(lambda row: ' '.join(row), axis=1)
        return res

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    profiles = get_profile_text()
    vectorizer = CountVectorizer()
    #transformer = TfidfTransformer()
    cntTf = vectorizer.fit_transform(profiles)

    n_topics = 5
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    learning_offset=50.,
                                    random_state=666)
    docres = lda.fit_transform(cntTf)
    print(docres.shape)
    return pd.DataFrame(docres, columns=[f'profile_lda_{i}' for i in range(n_topics)])


def get_feature_partition(cut_begin=48, cut_end=60):
    feature = get_feature( )
    feature = feature.loc[(feature['day'] >= cut_begin)
                          & (feature['day'] <= cut_end)
                          & (feature['click_mode'] >=0 )
                            ]
    sample =feature.click_mode.value_counts().sort_index().to_frame()
    sample = sample/sample.sum()
    sample['type']=f'sample_{cut_begin}_{cut_end}'
    return sample


#
# def get_train_test_zero():
#     feature = get_feature()  # .fillna(0)
#     logger.info(f'Remove simple zero case:{len(feature.loc[feature.o_seq_0 == 0])}')
#     feature = feature.loc[feature.o_seq_0 > 0]
#
#     def convert(val):
#         if val >= 1:
#             return 0
#         elif val == 0:
#             return 1
#         elif val == -1:
#             return -1
#         else:
#             return None
#
#     feature.click_mode = feature.click_mode.apply(lambda item: convert(item))
#
#     for precision in [6]:
#         feature[f'o_d_hash_{precision}'] = feature[f'o_d_hash_{precision}'].astype('category').cat.codes.astype(int)
#         feature[f'd_hash_{precision}'] = feature[f'd_hash_{precision}'].astype('category').cat.codes.astype(int)
#         feature[f'o_hash_{precision}'] = feature[f'o_hash_{precision}'].astype('category').cat.codes.astype(int)
#
#     remove_list = ['o_d_hash_5', 'd_hash_5', 'o_hash_5', 'plans',
#                    'o', 'd', 'label', 'req_time', 'click_time', 'date',
#                    'day', 'plan_time','plan_time_',
#                    #'s_pid_o_hash_m_per', 's_pid_d_hash_m_per',
#                      ]
#
#     remove_list.extend([col for col in feature.columns if 's_' in col])
#     remove_list.extend([col for col in [ f'{i}_transport_mode' for i in range(1, 12)]])
#
#     feature = feature.drop(remove_list, axis=1, errors='ignore')
#
#     logger.info((feature.shape, list(feature.columns)))
#
#     for col, type_ in feature.dtypes.sort_values().iteritems():
#         if type_ not in ['int64', 'int16', 'int32', 'float64']:
#             logger.error(col, type_)
#
#     #feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]
#
#     train_data = feature.loc[(feature.click_mode >= 0)]
#
#     X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]
#
#     return train_data, X_test
#

@timed()
def get_train_test(enhance_level):
    """
    train:500000, online:94358
    :param drop_list:
    :return:
    """
    feature = get_feature()  # .fillna(0)
    # logger.info(f'Remove simple zero case:{len(feature.loc[feature.o_seq_0 == 0])}')
    # feature = feature.loc[feature.o_seq_0 > 0]
    #There 2 days only have zero mode
    #feature = feature[~feature.day.isin([8, 35])]

    #feature = resample_train()

    for precision in [6]:
        feature[f'o_d_hash_{precision}'] = feature[f'o_d_hash_{precision}'].astype('category').cat.codes.astype(int)
        feature[f'd_hash_{precision}'] = feature[f'd_hash_{precision}'].astype('category').cat.codes.astype(int)
        feature[f'o_hash_{precision}'] = feature[f'o_hash_{precision}'].astype('category').cat.codes.astype(int)


    #feature = feature.drop(drop_list, axis=1, errors='ignore')


    #feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]

    train_data = feature.loc[(feature.click_mode >= 0)]
    train_data = train_data.loc[train_data.en_label <= enhance_level ]

    X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

    #logger.info((train_data.shape, list(train_data.columns)))

    return train_data, X_test

@timed()
#@lru_cache()
@file_cache()
def get_feature(ratio_base=0.1, group=None, ):
    query = get_query()
    plans = get_plans()

    #plan_cat = get_plan_cat()
    #plans.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plans.columns]

    click = get_click()


    #del plans['plan_time']


    query = pd.merge(query, plans, how='left', on='sid')

    #Fix the distance precision issue
    query.loc[query.sphere_dis <= 10, 'sphere_dis' ] = query.loc[query.sphere_dis <= 10, 'ps_distance_min']

    for direct in ['o']:
        precision = 6
        gp_level = [f'{direct}_hash_{precision}']
        geo_hash = get_geo_percentage(query, direct, gp_level, )
        query = pd.merge(query, geo_hash, how='left', on=gp_level)

    # for i in range(1, 12):
    #     query[f'{i}_distance_ratio'] = query[f'{i}_distance']/query['raw_dis']

    #query = pd.merge(query, plan_cat, how='left', on='sid')

    #if group is not None and 'profile' in group:
    profile = get_profile()
    query = pd.merge(query, profile, how='left', on='pid')

    #statistics, information
    stat = get_stati_feature_pid()
    query = pd.merge(query, stat, how='left', on='pid')
    logger.info('Finish merge stat feature')

    query = pd.merge(query, click, how='left', on='sid')
    query.loc[(query.label == 'train') & pd.isna(query.click_mode) & (query.o_seq_0 > 0), 'click_mode']  = 0
    query.loc[(query.label == 'train') & pd.isna(query.click_mode) & pd.isna(query.o_seq_0), 'click_mode'] = 0 # -2

    query.click_mode = query.click_mode.fillna(-1)
    query.click_mode = query.click_mode.astype(int)
    query.pid        = query.pid.astype(int)
    logger.info('Finish merge click feature')

    query = query.set_index('sid').sort_index()
    logger.info('Finish set index')
    query = query.fillna(0)
    logger.info('Finish fillna')
    return query


from math import radians, atan, tan, sin, acos, cos


def getDistance(latA, lonA, latB, lonB):
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
        return 0.0000001



def get_convert_profile_click_percent(feature):
    col = sorted([item for item in feature.columns if item.endswith('transport_mode')])
    df_col = col.copy()
    gp_level  = ['pid']
    df_col.extend( gp_level )
    df_col.append( 'click_mode'  )
    # Click
    logger.info(f'Only base on day from 0 to {val_cut_point} to cal click percentage')
    click = feature.loc[:, df_col].copy()
    for sn, cur_col in enumerate(tqdm(col, 'Click sum') ):
        #print(cur_col)
        click[cur_col] = click.apply(lambda item: 1 if item[cur_col] > 0 and item[cur_col] == item.click_mode else 0,
                                     axis=1)

    # Ignore
    ignore = feature.loc[:, df_col].copy()
    for sn, cur_col in  enumerate(tqdm(col, 'Ignore sum') ):
        #print('ignore', cur_col)
        ignore[cur_col] = ignore.apply(
            lambda item: 1 if item[cur_col] > 0 and item[cur_col] != item['click_mode'] else 0, axis=1)

    profile = pd.DataFrame()
    for sn, cur_col in enumerate(tqdm(col, 'Cal percentage') ):
        #print(cur_col)
        click_total = click.groupby(gp_level)[cur_col].agg({f'click_p_{sn+1:02}': 'sum'})
        ignore_total = ignore.groupby(gp_level)[cur_col].agg({f'click_p_{sn+1:02}': 'sum'})
        percent = click_total / (click_total + ignore_total)
        # print(type(percent))
        profile = pd.concat([profile, percent], axis=1)  # .fillna(0)
    profile = profile.fillna(0).reset_index()
    #profile.day = profile.day+7
    return profile


def get_convert_recommend(feature):
    new_fea = feature.copy()
    new_fea = new_fea.loc[new_fea.click_mode >=0 ]
    new_fea.click_mode = new_fea.click_mode == new_fea.o_seq_0

    gp_col = 'o_seq_0'
    new_fea = new_fea.groupby(gp_col).click_mode.agg(['sum','count'])
    new_fea[f'con_{gp_col}'] = new_fea['sum']/ new_fea['count']
    return new_fea.iloc[:, -1].reset_index()

@timed()
def extend_split_feature(train, val, test, drop_list=[]):
    train_index = train.index
    val_index = val.index
    test_index = test.index
    # profile_click =  get_convert_profile_click_percent(train)
    #
    # train = pd.merge( train, profile_click, how='left', on='pid' )
    # val   = pd.merge( val,   profile_click, how='left', on='pid')
    # test  = pd.merge( test,  profile_click, how='left', on='pid')

    # recommend = get_convert_recommend(train)
    # train = pd.merge(  train, recommend,  how='left', on='o_seq_0')
    # val   = pd.merge(  val,   recommend, how='left', on='o_seq_0')
    # test  = pd.merge(  test,  recommend, how='left', on='o_seq_0')

    del train['click_mode']
    del val['click_mode']


    remove_list = ['o_d_hash_5', 'd_hash_5', 'o_hash_5', 'plans',
                   'o', 'd', 'label', 'req_time', 'click_time', 'date',
                   'day', 'plan_time','sid', 'dummy_sid',
                   'req_time','plan_time','click_time', 'en_label', 'en_lable',
                   'sphere_dis','o_seq_7',

                   #'s_pid_o_hash_m_per', 's_pid_d_hash_m_per',
                     ]

    remove_list.extend(drop_list)

    remove_list.extend([col for col in train.columns if col.startswith('s_')])

    #pe_eta_price and so on
    remove_list.extend([col for col in train.columns if '_pe_' in col])

    remove_list.extend([col for col in train.columns if 'ps_' in col])

    #remove_list.extend([col for col in [ f'{i}_transport_mode' for i in range(1, 12)]])
    logger.info(f'Final remove list:{remove_list}')

    train = train.drop(remove_list, axis=1, errors='ignore')
    val = val.drop(remove_list, axis=1, errors='ignore')
    test = test.drop(remove_list, axis=1, errors='ignore')

    for col, type_ in train.dtypes.sort_values().iteritems():
        if type_ not in ['int64', 'int16', 'int32', 'float64']:
            logger.error((col, type_))

    train.index = train_index
    val.index = val_index
    test.index = test_index
    logger.info(f'extend_split_feature Train:{train.shape}, val:{val.shape}, test:{test.shape}, col_list:{list(train.columns)}')
    return train, val, test



if __name__ == '__main__':
    deep = get_plan_original_deep() #main
    #get_plan_original_deep() # main
    get_plan_original_wide()
    get_feature()
    """
    nohup python -u  core/feature.py   > feature.log  2>&1 &
    """

