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
    """
    get the display model sequence
    :return:
    """
    def get_model(plans):
        plans = json.loads(plans)
        res = {}
        if plans:
            for sn, plan in enumerate(plans):
                res[str(sn)] = plan['transport_mode']
        return Series(res)

    res_list = []
    for plan_file in ['train_plans.csv', 'test_plans.csv']:
        original_plan = get_original(plan_file)
        original_plan = original_plan.set_index(['sid'])
        del original_plan['plan_time']

        model_seq = original_plan.plans.apply(lambda item: get_model(item))

        model_seq.columns = [f'o_seq_{item}' for item in  model_seq.columns]

        model_seq = model_seq.fillna(0)

        res_list.append(model_seq.astype(int) )

    return pd.concat(res_list)

@file_cache()
def get_plan_mini(model=1, plan_file='test_plans.csv', ):
    """
    split the jasn to 11 model group
    :param plan_file:
    :param model:
    :return:
    """
    original_plan = get_original(plan_file)
    def get_model(plans, model):
        plans = json.loads(plans)
        if plans:
            for sn, plan in enumerate(plans):
                if plan['transport_mode'] == model:
                    #plan['order'] = sn
                    if plan['price'] is None or len(str(plan['price'])) == 0:
                        plan['price'] = 0
                    return Series(plan).astype(int)
        return Series()

    # model=1
    sigle_model = original_plan.plans.apply(lambda item: get_model(item, model))

    sigle_model['price'] = sigle_model['price'].fillna(0)  # .astype(float)

    sigle_model = sigle_model.fillna(0).astype(int)


    if 'distance' not in sigle_model.columns:
        logger.warning(f'No plan is found for model:{model}')
        return pd.DataFrame()
    col_list = plan_items

    mini_plan = original_plan.loc[:, ['sid']]

    mini_plan[col_list] = sigle_model[col_list]

    mini_plan = mini_plan.set_index(['sid'])

    mini_plan['pe_eta_price'] = mini_plan['eta']/mini_plan['price']
    mini_plan['pe_dis_price'] = mini_plan['distance']/mini_plan['price']
    mini_plan['pe_dis_eta']   = mini_plan['distance']/mini_plan['eta']

    mini_plan['pe_price_eta'] = mini_plan['price']/mini_plan['eta']
    mini_plan['pe_price_dis'] = mini_plan['price']/mini_plan['distance']
    mini_plan['pe_eta_dis']   = mini_plan['eta']/mini_plan['distance']

    #mini_plan['transport_mode'] = mini_plan['transport_mode']//model

    mini_plan.columns = [[str(model)] * len(mini_plan.columns), mini_plan.columns]
    return mini_plan


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
        col_list = [col for col in plan.columns if col[1] == item]
        plan_percent = plan.loc[:, col_list].copy()
        total = plan_percent.max(axis=1)
        for col in plan_percent:
            plan_percent[(str(col[0]), f'{col[1]}_max_p')] = round(plan_percent[col] / total, 4)
            del plan_percent[col]

        res_list.append(plan_percent)
    res = pd.concat(res_list, axis=1)

    # res.columns.set_levels([ f'{item[1]}_p' for item in res.columns ],level=1,inplace=True)
    # res.columns = [ (item[0], f'{item[1]}_p') for item in res.columns]
    res = res.sort_index(axis=1, level=1)
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

    plan.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plan.columns]

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

@file_cache()
def get_plan_original_deep():
    plan_list = []
    for plan_file in ['train_plans.csv', 'test_plans.csv']:
        original_plan = get_original(plan_file)
        for sn, row in tqdm(original_plan.iterrows(), f'convert {plan_file}'):
            plans = json.loads(row.plans)
            for single_plan in plans:
                single_plan['sid'] = row.sid
                plan_list.append(single_plan)
    res = pd.DataFrame(plan_list)
    return res.fillna('0').replace({'':'0'}).astype(int)



@timed()
#@lru_cache()
@file_cache()
def get_plan_original_wide():
    res_list = []
    for file in ['train_plans.csv', 'test_plans.csv']:
        base = get_original(file)
        base = base.set_index(['sid'])
        #del base['plan_time']
        from multiprocessing import Pool as ThreadPool  # 进程
        from functools import partial
        get_plan_mini_ex = partial(get_plan_mini, plan_file= file)

        pool = ThreadPool(6)
        plan_list = pool.map(get_plan_mini_ex, tqdm(range(1, 12)), chunksize=1)

        # if wide:
        plan_list.append(base)

        plan_part = pd.concat(plan_list, axis=1)
        res_list.append(plan_part)
        # else:
        #     for df in tqdm(plan_list):
        #         df.columns = df.columns.droplevel(0)
        #         df = df.loc[df.transport_mode>0]
        #         res_list.append(df)

    return pd.concat(res_list, axis=0)



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

    plans = get_plan_original_deep() #get_plan_stati_feature_sid

    for col in plan_items_mini:
        tmp = plans.groupby('sid')[col].agg(['min', 'max', 'mean', 'std']).add_prefix(f'ps_{col}_')
        res_list.append(tmp)

    tmp = plans.groupby('sid')['transport_mode'].agg(['count','nunique']).add_prefix(f'ps_transport_mode_')
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
    feature = get_feature( ).copy()
    sample = feature.loc[(feature['day'] >= cut_begin)
                          & (feature['day'] <= cut_end)
                          & (feature['click_mode'] >=0 )
                            ]
    sample =sample.click_mode.value_counts().sort_index().to_frame()
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
def get_train_test():
    """
    train:500000, online:94358
    :param drop_list:
    :return:
    """
    feature = get_feature().copy()  # .fillna(0)
    # logger.info(f'Remove simple zero case:{len(feature.loc[feature.o_seq_0 == 0])}')
    # feature = feature.loc[feature.o_seq_0 > 0]
    #There 2 days only have zero mode
    #feature = feature[~feature.day.isin([8, 35])]

    #feature = resample_train()

    for precision in [6]:
        feature[f'o_d_hash_{precision}'] = feature[f'o_d_hash_{precision}'].astype('category').cat.codes.astype(int)
        feature[f'd_hash_{precision}'] = feature[f'd_hash_{precision}'].astype('category').cat.codes.astype(int)
        feature[f'o_hash_{precision}'] = feature[f'o_hash_{precision}'].astype('category').cat.codes.astype(int)


    # remove_list = ['o_d_hash_5', 'd_hash_5', 'o_hash_5', 'plans',
    #                'o', 'd', 'label', 'req_time', 'click_time', 'date',
    #                'day', 'plan_time','sphere_dis','en_label', 'time_gap',
    #
    #                #'s_pid_o_hash_m_per', 's_pid_d_hash_m_per',
    #                  ]
    #
    # remove_list.extend(drop_list)
    # remove_list.extend([col for col in feature.columns if col.startswith('s_')])
    #
    # #pe_eta_price and so on
    # remove_list.extend([col for col in feature.columns if '_pe_' in col])
    #
    # remove_list.extend([col for col in feature.columns if 'ps_' in col])
    #
    # #remove_list.extend([col for col in [ f'{i}_transport_mode' for i in range(1, 12)]])
    # logger.info(f'Final remove list:{remove_list}')
    # feature = feature.drop(remove_list, axis=1, errors='ignore')
    # #feature = feature.drop(drop_list, axis=1, errors='ignore')
    #
    # for col, type_ in feature.dtypes.sort_values().iteritems():
    #     if type_ not in ['int64', 'int16', 'int32', 'float64']:
    #         logger.error(col, type_)

    #feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]

    train_data = feature.loc[(feature.click_mode >= 0)]

    X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

    logger.info((len(train_data), len(X_test.columns) , list(X_test.columns)))

    return train_data, X_test


@lru_cache()
@file_cache()
def get_feature(ratio_base=0.1, group=None, ):
    query = get_query()
    plans = get_plans()

    #plan_cat = get_plan_cat()
    #plans.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plans.columns]

    click = get_click()


    #del plans['plan_time']


    query = pd.merge(query, plans, how='left', on='sid')

    time_gap = (pd.to_datetime(query.plan_time) - pd.to_datetime(query.req_time)).dt.total_seconds()
    query['time_gap'] = time_gap.where(time_gap >= 0, -1)

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
    query.pid = query.pid.astype(int)
    logger.info('Finish merge stat feature')

    query = pd.merge(query, click, how='left', on='sid')
    query.loc[(query.label == 'train') & pd.isna(query.click_mode) & (query.o_seq_0 > 0), 'click_mode']  = 0
    query.loc[(query.label == 'train') & pd.isna(query.click_mode) & pd.isna(query.o_seq_0), 'click_mode'] = 0 # -2
    query.click_mode = query.click_mode.fillna(-1).astype(int)


    #Make click mode the last col
    click_mode = query.click_mode
    del query['click_mode']
    query['click_mode'] = click_mode

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


def get_convert_recommend(feature, gp_col = ['o_seq_0']):
    new_fea = feature.copy()
    new_fea = new_fea.loc[new_fea.click_mode >=0 ]
    new_fea.click_mode = new_fea.click_mode == new_fea.o_seq_0

    new_fea = new_fea.groupby(gp_col).click_mode.agg(['sum','count'])
    new_fea[f'conv_ratio'] = new_fea['sum']/ new_fea['count']
    new_fea =  new_fea.add_prefix('rec_conv_'+'_'.join(gp_col)+'_')
    return new_fea.reset_index()

def sample_ex(df:pd.DataFrame, frac):
    res_list = [df]*int(frac)
    res_list.append(df.sample(frac=frac%1, random_state=2019))
    return pd.concat(res_list)

@timed()
def extend_split_feature(df, trn_idx, val_idx ,  X_test, drop_list,):
    val_x = df.iloc[val_idx, :-1]
    val_y = df.iloc[val_idx].click_mode

    train = df.iloc[trn_idx]

    # stat_col = ['pid','o_seq_0']
    # conv = get_convert_recommend(train, stat_col)
    # val_x = pd.merge(val_x, conv, how='left', on=stat_col)
    # train = pd.merge(train, conv, how='left', on=stat_col)
    # X_test = pd.merge(X_test, conv,how='left', on=stat_col)
    #

    click_mode = train.click_mode
    del train['click_mode']
    train['click_mode'] = click_mode

    # for mode, frac in dict(enhance).items():
    #     base = train.loc[train.click_mode==mode]
    #     add_df = sample_ex(base, frac)
    #     logger.info(f'Enhance mode:{mode}, append {len(add_df)} records with frac:{frac} and base:{base.shape}')
    #     train = train.append(add_df)

    train_x = train.iloc[:, :-1]
    train_y = train.click_mode

    train_x = remove_col(train_x, drop_list).fillna(0)
    val_x   = remove_col(val_x, drop_list).fillna(0)
    X_test = remove_col(X_test, drop_list).fillna(0)


    for col, type_ in val_x.dtypes.sort_values().iteritems():
        if type_ not in ['int64', 'int16', 'int32', 'float64']:
            logger.error((col, type_))


    logger.info(f'extend_split_feature Train:{train_x.shape}, val:{val_x.shape}, col_list:{list(val_x.columns)}')
    #Drop end

    return train_x, train_y, val_x, val_y, X_test


def remove_col(train, drop_list):
    # Drop begin
    remove_list = ['o_d_hash_5', 'd_hash_5', 'o_hash_5', 'plans',
                   'o', 'd', 'label', 'req_time', 'click_time', 'date',
                   'day', 'plan_time', 'sphere_dis', 'en_label', 'time_gap',
                   '10_eta',

                   # 's_pid_o_hash_m_per', 's_pid_d_hash_m_per',
                   ]
    remove_list.extend(drop_list)
    remove_list.extend([col for col in train.columns if col.startswith('s_')])
    # pe_eta_price and so on
    remove_list.extend([col for col in train.columns if '_pe_' in col])
    remove_list.extend([col for col in train.columns if 'ps_' in col])
    # remove_list.extend([col for col in [ f'{i}_transport_mode' for i in range(1, 12)]])
    logger.info(f'Final remove list:{remove_list}')
    train = train.drop(remove_list, axis=1, errors='ignore')
    return train


if __name__ == '__main__':
    get_plan_original_deep() # main
    get_plan_original_wide()
    # get_feature()
    """
    nohup python -u  core/feature.py   > feature.log  2>&1 &
    """

