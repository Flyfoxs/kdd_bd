import sys
import os
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import f1_score

from core.feature import get_feature_core
from core.split import manual_split

warnings.filterwarnings("ignore")

from tqdm import tqdm



from file_cache.utils.util_pandas import *
from functools import lru_cache

#begin = '2018-11-03' # '2018-11-17'
#end = '2018-12-01'
train_cnt =0

bin_cnt = 2

# df = pd.DataFrame(columns=['paras', 'score'])
def get_best_paras(df: pd.DataFrame, pid=1):
    df = df.sort_values('adj_score', ascending=False)
    #logger.debug(f'cur best:{df.score.values[0]}, {df.paras.values[0]}')
    return df.loc[df.pid==pid].paras.values[:1]


def check_insert(df: pd.DataFrame, paras, pid=1):
    if not isinstance(paras, tuple):
        paras = tuple(list(np.array(paras).round(3)))


    # if  any(np.isin(paras,df.paras.values)):
    exist = df.loc[(df.paras==paras) & (df.pid==pid) ]

    if len(exist) == 1:
        # logger.info(f'ALready find {len(exist)} score:{exist.raw_score.max()}=>{exist.adj_score.min()} for {pid}, {paras}')
        pass
    elif len(exist) > 1:
        print('====', exist)
        raise Exception('Duplicate para')
    else:  ###
        # print('append', paras, type(paras))
        for pid in range(1):
            df = df.append({'paras': paras, 'pid':pid}, ignore_index=True)
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
def find_best_para():
    paras_df = pd.DataFrame(columns=['paras',  'raw_score','adj_score', 'val_score', 'pid'])

    #initial paras
    if len(paras_df) == 0:
        for pid in range(1):
            paras_df = paras_df.append({'paras': (1.12, 0.9, 0.8, 2.43, 2.9, 0.8, 1.85, 0.93, 1.52, 0.91, 1.33, 1.08)
                               , 'pid': pid}, ignore_index=True)

    global train_cnt

    partition_list = get_partition()
    train_partition =  partition_list[0]
    val_partition = partition_list[1]
    logger.info(f'folder#{pid}')
    train_cnt = len(train_partition)

    raw_score = f1_score(train_partition.click_mode.values, train_partition.iloc[:, :-1].idxmax(axis=1).astype(int), average='weighted')

    lr = 0.1
    for i in range(2000):
        for paras in get_best_paras(paras_df, pid):
            # print(type(paras), paras)
            paras = list(paras)
            paras_df = check_insert(paras_df, paras, pid)
            ##print('===', paras)
            for i in range(12):
                for times in range(1, 3):
                    new = paras.copy()
                    new[i] = new[i] + lr * times
                    paras_df = check_insert(paras_df, new, pid)

                    new = paras.copy()
                    new[i] = new[i] - lr * times
                    paras_df = check_insert(paras_df, new, pid)

                    # print('======',paras_df.shape)

        todo = paras_df.loc[(pd.isna(paras_df.adj_score))]

        if len(todo) == 0 and lr <= 0.01:
            break
        elif len(todo) == 0:
            lr = 0.01

        ex_todo = paras_df.loc[(pd.isna(paras_df.adj_score))]
        for sn, row in tqdm(ex_todo.iterrows(),
                            f'{len(todo)}/{len(paras_df)} paras need to process with {lr}'):



            paras_df.loc[sn, 'raw_score'] = raw_score
            paras_df.loc[sn, 'adj_score'] = evaluate_res_ratio(train_partition.copy(), row['paras'])
            paras_df.loc[sn, 'val_score'] = evaluate_res_ratio(val_partition.copy(), row['paras'])
            if len(train_partition) == 0:
                break
            #print(f"{score}:{row['paras']}")

        paras_df = paras_df.sort_values('adj_score', ascending=False).reset_index(drop=True)
        para_file = f'./output/stacking/paras_300.h5'
        paras_df.to_hdf(para_file, 'para')
    logger.info(f'Para save to file:{para_file}')
    return paras_df




def gen_sub_file(input_file, paras, adj_score, raw_score):
    sub = pd.read_hdf(input_file, 'test')

    query = get_feature_core()[['city']]
    sub = sub.join(query)


    #sub_file = f'./output/sub/ad_avg_{adj_score:6.5f}_{raw_score:6.5f}_{begin}_{train_cnt}.csv'

    sub_file = f'./output/sub/ad_para_avg.csv'

    logger.info(f'There are {len(sub)} record will be adjust ')

    for i in range(12):
        sub.loc[:, str(i)] = sub.loc[:, str(i)] * paras[i]

    sub['recommend_mode'] = sub.iloc[:, :12].idxmax(axis=1)

    import csv
    #sub.index = pd.Series(sub.index).apply(lambda val: val.split('-')[-1])
    sub[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL)
    logger.info(f'>>>>Sub file save to {sub_file}')



from functools import reduce


def merge_file(input_list):
    for key in ['train', 'test']:
        df_list = [pd.read_hdf(file, key) * weight for file, weight in input_list]

        # product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
        df = reduce((lambda x, y: x + y), df_list)
        df = df / len(input_list)
    return df


@lru_cache()
def get_partition():
    train = pd.read_hdf('./output/stacking/avg.h5', 'train')

    feature = get_feature_core()
    feature = feature.loc[pd.to_datetime(feature.req_time.dt.date) >= pd.to_datetime('2018-11-10')]

    train = train.loc[train.index.isin(feature.sid)]

    train.click_mode = train.click_mode.astype(int)
    folds = manual_split()
    split_fold = folds.split(train, True, n_splits=bin_cnt)

    partitions = []

    for fold_, (trn_idx, val_idx) in enumerate(tqdm(split_fold, 'Kfold')):
        partitions.append(train.iloc[val_idx])

    return partitions

if __name__ == '__main__':
    find_best_para()
    # """
    # The stack_file require the format as below:
    # train:(13 columns): 0,1,..11,click_mode(不是 recomm_mode,而是原始label)
    # test: (12 columns): 0,1,..11
    #
    # index is sid, and DF is sorted by index asc
    # """
    #
    #
    # input_file = './output/stacking/L_True_2000000_649_0.66997_1035_1524.h5'
    #
    # adj = pd.read_hdf(input_file, 'train')
    #
    # #partition = get_partition()
    #
    # adj = adj.loc[adj.index.isin(partition.index)]
    #
    # raw_score = f1_score(adj.click_mode.values, adj.iloc[:, :12].idxmax(axis=1).astype(int), average='weighted')
    #
    # res_list = []
    #
    # para_file = f'{input_file}.para_bin3.h5'
    #
    # old_len = len(adj)
    # for bin_id in tqdm(partition.bin_id.drop_duplicates()):
    #         partition_cur = partition.loc[partition.bin_id==bin_id]
    #         partition_cur = adj.loc[adj.index.isin(partition_cur.index)]
    #         #print(f'one_day shape:{one_day.shape}')
    #         logger.info(f'only keep {len(partition_cur)} records from {old_len} records for: {bin_id}')
    #
    #         paras_df = find_best_para(partition_cur.copy())
    #         paras_df['count']  = len(partition_cur)
    #         paras_df['bin_id'] = bin_id
    #
    #         best = paras_df.iloc[0]
    #         logger.info(f'\nThe best result for {len(partition_cur)}#{bin_id}#{len(paras_df)}, {paras_df.raw_score.max()} {paras_df.adj_score.max()}, {list(best.paras)}')
    #         res_list.append(paras_df)
    # pd.concat(res_list).to_hdf(para_file, 'paras')
    # logger.info(f'para save to file:{para_file}')
    #
    # adj = get_adj_df(adj, para_file)
    #
    # adj_score = f1_score(adj.click_mode.values, adj.iloc[:, :12].idxmax(axis=1).astype(int), average='weighted')
    #
    #
    # gen_sub_file(input_file, para_file, adj_score, raw_score )

"""


nohup python -u  core/adj_search_bins.py   > adj_search_split.log 2>&1 &
"""
