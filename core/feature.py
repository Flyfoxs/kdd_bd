import sys
import os
import pandas as pd
import numpy as np

from core.config import *
from tqdm import tqdm


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
def get_plan_mini(plan_file, model):
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
    mini_plan.columns = [[model] * len(mini_plan.columns), mini_plan.columns]
    return mini_plan


@lru_cache()
def get_original(file):
    return pd.read_csv(f'{input_folder}/{file}', dtype=type_dict)


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
            plan_percent[(col[0], f'{col[1]}_p')] = round(plan_percent[col] / total, 4)
            del plan_percent[col]

        res_list.append(plan_percent)
    res = pd.concat(res_list, axis=1)

    # res.columns.set_levels([ f'{item[1]}_p' for item in res.columns ],level=1,inplace=True)
    # res.columns = [ (item[0], f'{item[1]}_p') for item in res.columns]
    res = res.sort_index(axis=1, level=1)
    return res

@timed()
def get_plans():
    """
    3, 5, 6 price is 0
    :return:
    """
    plan = get_plan_original()
    #res.plan_time = pd.to_datetime(res.plan_time)

    plan_pg = get_plan_percentage()
    plan[plan_pg.columns] = plan_pg[plan_pg.columns]

    seq = get_plan_model_sequence()
    plan[seq.columns] = seq[seq.columns]
    #plan = pd.merge(plan,seq, how='left', on='sid')


    return plan.reset_index()

@timed()
@file_cache()
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

    # day_ = pd.to_datetime(train.req_time).dt.date
    # day_ = day_ - min(day_)
    # day_ = day_.dt.days
    # train['gp'] = pd.qcut(day_, 5).cat.codes.values
    #

    test = get_original('test_queries.csv')
    test['label'] = 'test'
    # test['gp'] = -1

    train_query = pd.concat([train, test])
    train_query.pid = train_query.pid.fillna(0)
    train_query.pid = train_query.pid.astype(int)
    train_query.req_time = pd.to_datetime(train_query.req_time)
    #train_query['date'] = train_query.req_time.dt.date
    train_query['day'] = train_query.req_time.dt.day
    train_query['weekday'] = train_query.req_time.dt.weekday
    train_query['hour'] = train_query.req_time.dt.hour
    train_query['weekend'] = train_query.weekday // 5

    train_query['o0'] = train_query.o.apply(lambda item: item.split(',')[0]).astype(float)
    train_query['o1'] = train_query.o.apply(lambda item: item.split(',')[1]).astype(float)

    train_query['d0'] = train_query.d.apply(lambda item: item.split(',')[0]).astype(float)
    train_query['d1'] = train_query.d.apply(lambda item: item.split(',')[1]).astype(float)

    precision = 6
    train_query[f'o_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.o1, row.o0, precision=precision),
                                                           axis=1)

    train_query[f'd_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.d1, row.d0, precision=precision),
                                                           axis=1)

    train_query[f'o_d_hash_{precision}'] = train_query[f'o_hash_{precision}'] + '_' + train_query[f'd_hash_{precision}']

    return train_query





def get_click():
    return get_original('train_clicks.csv')

@timed()
@file_cache()
def get_feature(group=None):
    query = get_query()
    plans = get_plans()
    click = get_click()


    del plans['plan_time']



    query = pd.merge(query, plans, how='left', on='sid')

    if group is not None and 'profile' in group:
        profile = get_original('profiles.csv').astype(int)
        query = pd.merge(query, profile, how='left', on='pid')

    query = pd.merge(query, click, how='left', on='sid')
    query.loc[(query.label=='train') & pd.isna(query.click_mode), 'click_mode'] = 0


    remove_list = ['o','d', 'label', 'req_time','click_time', 'day']
    query = query.drop(remove_list,axis=1,errors='ignore')

    query.click_mode = query.click_mode.fillna(-1)
    query.click_mode = query.click_mode.astype(int)
    query.pid        = query.pid.astype(int)

    query['o_d_hash_6'] = query['o_d_hash_6'].astype('category').cat.codes
    query['d_hash_6']   = query['d_hash_6'].astype('category').cat.codes
    query['o_hash_6']   = query['o_hash_6'].astype('category').cat.codes
    query = query.set_index('sid')
    return query.fillna(0)



if __name__ == '__main__':
    get_feature(('profile'))

    get_feature(None)

