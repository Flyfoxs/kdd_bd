import sys
import os
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

from tqdm import tqdm



from file_cache.utils.util_pandas import *
from functools import lru_cache

begin = '2018-11-03' # '2018-11-17'
end = '2018-12-01'
train_cnt =0


# df = pd.DataFrame(columns=['paras', 'score'])
def get_best_paras(df: pd.DataFrame):
    if len(df) == 0:
        return [
            [1.12, 0.9, 0.8, 2.43, 2.9, 0.8, 1.85, 0.93, 1.52, 0.91, 1.33, 1.08],
            [1.12, 0.9, 0.7, 2.22, 2.92, 0.8, 1.63, 0.83, 1.3, 0.77, 1.21, 0.89]
                ]
        # return []
        # return [np.ones(12)]
    else:  ##
        df = df.sort_values('score', ascending=False)
        logger.debug(f'cur best:{df.score.values[0]}, {df.paras.values[0]}')
        return df.paras.values[:1]


def check_insert(df: pd.DataFrame, paras):
    if not isinstance(paras, tuple):
        paras = tuple(list(np.array(paras).round(3)))

    # print(paras)
    # if  any(np.isin(paras,df.paras.values)):
    if any(df.paras.isin([paras])):
        # print(len(df.paras.isin([paras])))
        # logger.info(f'Already existing {len(df)},  {df.loc[df.paras == paras, "score"]}, for {paras}')
        pass
    else:  ###
        # print('append', paras, type(paras))
        df = df.append({'paras': paras}, ignore_index=True)

        # print(len(df))
    return df


def evaluate_res_ratio(adj, p):
    #print('before',adj.sum().sum())
    for i in range(12):
        adj.iloc[:, i] = adj.iloc[:, i] * p[i]

    #print('new',adj.sum().sum())
    adj['recommend_mode'] = adj.iloc[:, :12].idxmax(axis=1)
    score_after = f1_score(adj.click_mode.astype(int), adj.recommend_mode.astype(int), average='weighted')
    return score_after


# file = './output/stacking/L_0.68018_0914_1667.h5'
@timed()
def find_best_para(adj):
    paras_df = pd.DataFrame(columns=['paras', 'score'])
    global train_cnt
    train_cnt = len(adj)

    raw_score = f1_score(adj.click_mode.values, adj.iloc[:, :-1].idxmax(axis=1).astype(int), average='weighted')

    lr = 0.1
    for i in range(2000):
        for paras in get_best_paras(paras_df):
            # print(type(paras), paras)
            paras = list(paras)
            paras_df = check_insert(paras_df, paras)
            ##print('===', paras)
            for i in range(12):
                for times in range(1, 3):
                    new = paras.copy()
                    new[i] = new[i] + lr * times
                    paras_df = check_insert(paras_df, new)

                    new = paras.copy()
                    new[i] = new[i] - lr * times
                    paras_df = check_insert(paras_df, new)

                    # print('======',paras_df.shape)
        if len(paras_df.loc[pd.isna(paras_df.score)]) == 0 and lr <= 0.01:
            break
        elif len(paras_df.loc[pd.isna(paras_df.score)]) == 0:
            lr = 0.01

        for sn, row in tqdm(paras_df.loc[pd.isna(paras_df.score)].iterrows(),
                            f'{len(paras_df.loc[pd.isna(paras_df.score)])}/{len(paras_df)} paras need to process with {lr}'):
            score = evaluate_res_ratio(adj.copy(), row['paras'])
            paras_df.loc[sn, 'score'] = score
            if len(adj) == 0:
                break
            #print(f"{score}:{row['paras']}")

    paras_df = paras_df.sort_values('score', ascending=False)
        #print(paras_df.shape)

    paras_df['raw_score'] = raw_score
    paras_df.to_csv('./output/search_para.csv')
    #print(get_best_paras(paras_df))
    #print(paras_df.head(5))
    return paras_df


def gen_sub_file(input_file, paras, adj_score, raw_score):
    sub = pd.read_hdf(input_file, 'test')

    sub_file = f'./output/sub/adj_{adj_score:0.5f}_{raw_score:0.6f}_{begin}_{train_cnt}.csv'

    for i in range(12):
        sub.iloc[:, i] = sub.iloc[:, i] * paras[i]

    sub['recommend_mode'] = sub.idxmax(axis=1)

    import csv
    sub.index = pd.Series(sub.index).apply(lambda val: val.split('-')[-1])
    sub[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL)
    logger.info(f'>>>>Sub file save to {sub_file}')


from functools import reduce


# input_list = [
#     ('./output/stacking/L_500000_268_0.67818_0439_1462.h5', 1),
#
#     ('./output/stacking/L_500000_190_0.67825_0534_1613.h5', 1), ]
#

def merge_file(input_list):
    for key in ['train', 'test']:
        df_list = [pd.read_hdf(file, key) * weight for file, weight in input_list]

        # product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
        df = reduce((lambda x, y: x + y), df_list)
        df = df / len(input_list)
    return df

def filter_by_data(df, req_time):
    from  core.feature import get_original
    query = get_original('train_queries_phase2.csv')
    query.req_time = pd.to_datetime(query.req_time)#.dt.date
    query = query.loc[ pd.to_datetime(query.req_time.dt.date) == pd.to_datetime(req_time) ]
    return df.loc[query.sid]

if __name__ == '__main__':
    """
    The stack_file require the format as below:
    train:(13 columns): 0,1,..11,click_mode(不是 recomm_mode,而是原始label)
    test: (12 columns): 0,1,..11

    index is sid, and DF is sorted by index asc
    """


    input_file='./output/stacking/L_2000000_480_0.66449_0629_1672.h5'

                              #'./output/stacking/L_2000000_336_0.65994_1539_2530.h5', #0.69506305
                              # './output/stacking/L_1500000_336_0.65318_1470_2220.h5',
                              # './output/stacking/L_1500000_336_0.65328_1129_2425.h5',

    adj = pd.read_hdf(input_file, 'train')
    old_len = len(adj)
    adj = adj.loc[adj.index.str.startswith('2-')]

    res_list = []

    for req_time in tqdm(pd.date_range(start='2018-11-10' , end='2018-11-10')):
            one_day = filter_by_data(adj, req_time)
            #print(f'one_day shape:{one_day.shape}')
            logger.info(f'only keep {len(one_day)} records from {old_len} records for: {req_time}')

            paras_df = find_best_para(one_day.copy())
            paras_df['count']    = len(one_day)
            paras_df['req_time'] = req_time

            logger.info(f'Estimate {len(paras_df)} paras for: {req_time}')

            res_list.append(paras_df)

    para_file = f'{input_file}.para.h5'
    pd.concat(res_list).to_hdf(para_file, 'paras')
    logger.info(f'para save to file:{para_file}')

"""


nohup python -u  core/adj_search.py   > adj_search.log 2>&1 &
"""
