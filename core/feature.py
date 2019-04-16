import sys
import os
import pandas as pd
import numpy as np


from tqdm import tqdm


from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache
from functools import lru_cache

from glob import glob
import json

# from core.feature import *
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

input_folder = './input/data_set_phase1'

from  pandas import Series


@timed()
def get_model_sequence():
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
        original_plan = original_plan.set_index('sid')


        model_seq = original_plan.plans.apply(lambda item: get_model(item))

        model_seq.columns = [f'o_seq_{item}' for item in  model_seq.columns]

        model_seq = model_seq.fillna(0)

        res_list.append(model_seq.astype(int).reset_index())

    return pd.concat(res_list)



# @timed()
@file_cache()
def get_mini_plan(plan_file, model):
    original_plan = get_original(plan_file)

    def get_model(plans, model):
        plans = json.loads(plans)
        if plans:
            for sn, plan in enumerate(plans):
                if plan['transport_mode'] == model:
                    plan['order']=sn
                    if plan['price'] is None or len(str(plan['price'])) == 0:
                        plan['price'] = 0
                    return Series(plan).astype(int)
        return Series()

    # model=1
    sigle_model = original_plan.plans.apply(lambda item: get_model(item, model))

    sigle_model['price'] = sigle_model['price'].fillna(0)#.astype(float)

    sigle_model = sigle_model.fillna(0).astype(int)

    if 'distance' not in sigle_model.columns:
        logger.warning(f'No plan is found for model:{model}')
        return pd.DataFrame()
    col_list = ['order', 'distance', 'eta', 'price', 'transport_mode']

    mini_plan = original_plan.loc[:, ['sid', 'plan_time']]
    # mini_plan = train_plan[['sid','plan_time']]
    mini_plan[[f'{item}_{model}' for item in col_list]] = sigle_model[col_list]


    return mini_plan


@lru_cache()
def get_original(file):
    return pd.read_csv(f'{input_folder}/{file}')

@timed()
def get_plans():
    """
    3, 5, 6 price is 0
    :return:
    """
    plan_list = []
    for plan_file in ['train_plans.csv', 'test_plans.csv']:
        base = get_mini_plan(plan_file, 1)
        for i in tqdm(range(2, 12)):
            tmp= get_mini_plan(plan_file, i)
            columns = tmp.columns[2:]
            #print(columns)
            base[columns] = tmp.loc[:, columns]
        plan_list.append(base)
    res = pd.concat(plan_list)
    res.plan_time = pd.to_datetime(res.plan_time)


    seq = get_model_sequence()
    res = pd.merge(res,seq, how='left', on='sid')
    return res


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
@lru_cache()
def get_feature():
    query = get_query()
    plans = get_plans()
    click = get_click()


    query = pd.merge(query, plans, how='left', on='sid')
    query = pd.merge(query, click, how='left', on='sid')

    return query



if __name__ == '__main__':
    get_feature()

