import sys
import os
import pandas as pd
import numpy as np

from core.config import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache
from functools import lru_cache

from glob import glob
import json


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
        original_plan = original_plan.set_index(['sid', 'plan_time'])


        model_seq = original_plan.plans.apply(lambda item: get_model(item))

        model_seq.columns = [f'o_seq_{item}' for item in  model_seq.columns]

        model_seq = model_seq.fillna(0)

        res_list.append(model_seq.astype(int) )

    return pd.concat(res_list)

@file_cache()
def get_plan_mini(plan_file='test_plans.csv', model=1):
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

    mini_plan = original_plan.loc[:, ['sid', 'plan_time']]

    mini_plan[col_list] = sigle_model[col_list]

    mini_plan = mini_plan.set_index(['sid', 'plan_time'])

    del mini_plan['transport_mode']

    mini_plan.columns = [[str(model)] * len(mini_plan.columns), mini_plan.columns]
    return mini_plan


@lru_cache()
def get_original(file):
    return pd.read_csv(f'{input_folder}/{file}', dtype=type_dict)


def get_profile_click_percent(feature):
    col = sorted([item for item in feature.columns if item.endswith('transport_mode')])
    df_col = col.copy()
    df_col.extend(['click_mode', 'pid','day'])
    # Click
    logger.info(f'Only base on day from 0 to {val_cut_point} to cal click percentage')
    click = feature.loc[:, df_col].copy()
    for sn, cur_col in enumerate(tqdm(col, 'Click sum') ):
        #print(cur_col)
        click[cur_col] = click.apply(lambda item: 1 if item[cur_col] > 0 and item[cur_col] == item['click_mode'] else 0,
                                     axis=1)

    # Ignore
    ignore = feature.loc[:, df_col].copy()
    for sn, cur_col in  enumerate(tqdm(col, 'Ignore sum') ):
        #print('ignore', cur_col)
        ignore[cur_col] = ignore.apply(
            lambda item: 1 if item[cur_col] > 0 and item[cur_col] != item['click_mode'] else 0, axis=1)

    profile = pd.DataFrame()
    for sn, cur_col in enumerate(tqdm(col, 'Cal percentage') ):
        print(cur_col)
        click_total = click.groupby(['pid','day'])[cur_col].agg({f'click_p_{sn+1:02}': 'sum'})
        ignore_total = ignore.groupby(['pid','day'])[cur_col].agg({f'click_p_{sn+1:02}': 'sum'})
        percent = click_total / (click_total + ignore_total)
        # print(type(percent))
        profile = pd.concat([profile, percent], axis=1)  # .fillna(0)
    profile = profile.fillna(0).reset_index()
    profile.day = profile.day+7
    return profile

def get_plan_summary():
    plan = get_plan_original()
    res_list = []
    for item in ['distance']:
        col_list = [col for col in plan.columns if col[1] == item]
        summary = plan.loc[:, col_list].copy()
        # max_ = summary.where(summary > 0).max(axis=1)
        # max_.name=f'{item}_max'
        # res_list.append(max_)
        min_  = summary.where(summary > 0).min(axis=1)

        min_ = pd.cut(min_,20).cat.codes
        min_.name = f'{item}_min_cat'
        res_list.append(min_)
    res = pd.concat(res_list, axis=1)
    res = res.sort_index(axis=1, level=1)
    return res


@timed()
@file_cache()
def get_plan_percentage_min():
    """
    Convert plan from amount/qty to percentage
    :param plan:
    :return:
    """
    plan = get_plan_original()
    res_list = []
    for item in plan_items:
        col_list = [col for col in plan.columns if col[1] == item]
        plan_percent = plan.loc[:, col_list].copy()
        total = plan_percent.where(plan_percent > 0).min(axis=1)
        for col in plan_percent:
            plan_percent[(str(col[0]), f'{col[1]}_min_p')] = round(plan_percent[col] / total, 4)
            del plan_percent[col]

        res_list.append(plan_percent)
    res = pd.concat(res_list, axis=1)

    # res.columns.set_levels([ f'{item[1]}_p' for item in res.columns ],level=1,inplace=True)
    # res.columns = [ (item[0], f'{item[1]}_p') for item in res.columns]
    res = res.sort_index(axis=1, level=1)
    return res


@timed()
@file_cache()
def get_plan_percentage():
    """
    Convert plan from amount/qty to percentage
    :param plan:
    :return:
    """
    plan = get_plan_original()
    res_list = []
    for item in plan_items:
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

@timed()
@file_cache()
def get_plans():
    """
    3, 5, 6 price is 0
    :return:
    """
    plan = get_plan_original()
    #res.plan_time = pd.to_datetime(res.plan_time)

    plan_pg = get_plan_percentage()
    plan[plan_pg.columns] = plan_pg[plan_pg.columns]

    # plan_pg = get_plan_percentage_min()
    # plan[plan_pg.columns] = plan_pg[plan_pg.columns]
    #

    seq = get_plan_model_sequence()
    plan[seq.columns] = seq[seq.columns]

    # summay = get_plan_summary()
    # plan[summay.columns] = summay[summay.columns]

    plan.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plan.columns]

    return plan.reset_index()

@timed()
@lru_cache()
# @file_cache(overwrite=True)
def get_plan_original():
    plan_list = []
    for plan_file in ['train_plans.csv', 'test_plans.csv']:
        base = get_plan_mini(plan_file, 1)
        for i in tqdm(range(2, 12)):
            tmp = get_plan_mini(plan_file, i)
            columns = tmp.columns#[2:]
            # print(columns)
            base[columns] = tmp.loc[:, columns]
        plan_list.append(base)
    plan = pd.concat(plan_list)
    return plan


@lru_cache()
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

    for precision in hash_precision:
        train_query[f'o_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.o1, row.o0, precision=precision),
                                                               axis=1)

        train_query[f'd_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.d1, row.d0, precision=precision),
                                                               axis=1)

        train_query[f'o_d_hash_{precision}'] = train_query[f'o_hash_{precision}'] + '_' + train_query[f'd_hash_{precision}']

    return train_query





def get_click():
    return get_original('train_clicks.csv')

def get_profile():
    profile = get_original('profiles.csv').astype(int)
    # p_len = 66
    # for i in range(p_len):
    #     for j in range(i + 1, p_len):
    #         new_p = profile[f'p{i}'] * profile[f'p{j}']
    #         ratio = sum(new_p) / new_p.count()
    #         if ratio > ratio_base:
    #             logger.info(f'p{i:02}_p{j:02}:{sum(new_p)} , {ratio}')
    #             profile[f'p{i:02}_p{j:02}'] = new_p
    # logger.info(f'The shape of profile is:{profile.shape}')
    return profile

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



@timed()
def resample_train(begin=54, end=60):
    feature = get_feature()
    feature = feature.loc[feature.click_mode>=0]
    gp = feature.click_mode.value_counts()
    gp = gp.loc[gp.index>=0].sort_index()
    base = gp.min()

    ratio = get_feature_partition(cut_begin=begin, cut_end=end)
    ratio = ratio.click_mode/ratio.click_mode.min()
    sample_count = round(ratio*base).astype(int)

    new_df = feature.loc[feature.click_mode==-1]

    for i in tqdm(range(0, 12), 'resample base on ratio'):
        cnt = sample_count.loc[i]
        tmp_df = feature.loc[feature.click_mode == i].sample(cnt)
        new_df = pd.concat([new_df, tmp_df])
    return new_df


def get_train_test(drop_list):
    #feature = get_feature()  # .fillna(0)

    feature = resample_train()

    remove_list = ['o', 'd', 'label', 'req_time', 'click_time', 'day', 'plan_time','plan_time_', ]
    feature = feature.drop(remove_list, axis=1, errors='ignore')
    feature = feature.drop(drop_list, axis=1, errors='ignore')

    logger.info((feature.shape, list(feature.columns)))

    for col, type_ in feature.dtypes.sort_values().iteritems():
        if type_ not in ['int64', 'int16', 'int32', 'float64']:
            logger.error(col, type_)

    feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]

    train_data = feature.loc[(feature.click_mode >= 0) & (feature.o_seq_0_ > 0)]

    X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

    return train_data, X_test

@timed()
#@lru_cache()
@file_cache()
def get_feature(ratio_base=0.1, group=None, ):
    query = get_query()
    plans = get_plans()
    #plans.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plans.columns]

    click = get_click()


    #del plans['plan_time']


    query = pd.merge(query, plans, how='left', on='sid')

    #if group is not None and 'profile' in group:
    profile = get_profile()
    query = pd.merge(query, profile, how='left', on='pid')

    query = pd.merge(query, click, how='left', on='sid')
    query.loc[(query.label == 'train') & pd.isna(query.click_mode) & (query.o_seq_0_ > 0), 'click_mode']  = 0
    query.loc[(query.label == 'train') & pd.isna(query.click_mode) & (query.o_seq_0_ == 0), 'click_mode'] = -2

    query.click_mode = query.click_mode.fillna(-1)
    query.click_mode = query.click_mode.astype(int)
    query.pid        = query.pid.astype(int)

    for precision in hash_precision:
        query[f'o_d_hash_{precision}'] = query[f'o_d_hash_{precision}'].astype('category').cat.codes
        query[f'd_hash_{precision}']   = query[f'd_hash_{precision}'].astype('category').cat.codes
        query[f'o_hash_{precision}']   = query[f'o_hash_{precision}'].astype('category').cat.codes
    query = query.set_index('sid')



    # profile_per = get_profile_click_percent(query)
    # query = pd.merge(query, profile_per, how='left', on=['pid','day'])
    #
    # #Make click_mode(label) is in the end
    # click_mode = query.click_mode.copy()
    # del query['click_mode']
    # query['click_mode'] = click_mode


    return query.fillna(0)



if __name__ == '__main__':


    get_feature()
    """
    nohup python -u  core/feature.py   > feature.log  2>&1 &
    """

