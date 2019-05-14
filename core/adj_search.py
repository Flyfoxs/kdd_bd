import sys
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from core.config import *
from core.feature import *
from core.train import *

from tqdm import tqdm



from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache
from functools import lru_cache

input_folder = './input/data_set_phase1'



# df = pd.DataFrame(columns=['paras', 'score'])
def get_best_paras(df: pd.DataFrame):
    if len(df) == 0:
        return [0.9, 1.0, 0.6, 1.2, 1.6, 0.7, 0.9, 0.9, 1.6, 0.5, 1.3, 1.1]
        #return [np.ones(12)]
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
        #logger.info(f'Already existing {len(df)},  {df.loc[df.paras == paras, "score"]}, for {paras}')
        pass
    else:  ###
        # print('append', paras, type(paras))
        df = df.append({'paras': paras}, ignore_index=True)

        # print(len(df))
    return df


def adjust_res_ratio(adj, p):
    for i in range(12):
        adj.iloc[:, i] = adj.iloc[:, i] * p[i]

    adj['recommend_mode'] = adj.iloc[:, :12].idxmax(axis=1)
    score_after = f1_score(adj.click_mode.astype(int), adj.recommend_mode.astype(int), average='weighted')
    return score_after


#file = './output/stacking/L_0.68018_0914_1667.h5'
@timed()
def find_best_para(file):
    df = pd.DataFrame(columns=['paras', 'score'])

    adj = pd.read_hdf(file, 'train')
    lr = 0.1
    for i in range(2000):
        for paras in get_best_paras(df):
            # print(type(paras), paras)
            paras = list(paras)
            df = check_insert(df, paras)
            ##print('===', paras)
            for i in range(12):
                for times in range(1, 3):
                    new = paras.copy()
                    new[i] = new[i] + lr * times
                    df = check_insert(df, new)

                    new = paras.copy()
                    new[i] = new[i] - lr * times
                    df = check_insert(df, new)

                    # print('======',df.shape)
        if len(df.loc[pd.isna(df.score)]) == 0 and lr <= 0.01:
            break
        elif len(df.loc[pd.isna(df.score)]) == 0:
            lr = 0.01

        for sn, row in tqdm(df.loc[pd.isna(df.score)].iterrows(),
                            f'{len(df.loc[pd.isna(df.score)])}/{len(df)} paras need to process with {lr}'):
            df.loc[sn, 'score'] = adjust_res_ratio(adj.copy(), row['paras'])

        df = df.sort_values('score', ascending=False)
        print(df.shape)

    df.to_csv('./output/search_para.csv')
    print(get_best_paras(df))
    print(df.head(5))
    return df.iloc[0].score, list(df.iloc[0].paras)

def gen_sub_file(input_file, paras, sub_file):
    sub = pd.read_hdf(input_file, 'test')
    for i in range(12):
        sub.iloc[:, i] = sub.iloc[:, i] * paras[i]

    sub['recommend_mode'] = sub.idxmax(axis=1)


    #vali_sub(sub)

    import csv
    sub[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL)
    logger.info(f'Sub file save to {sub_file}')


if __name__== '__main__':
    """
    The stack_file h5 format
    train(13 columns):0,1,..11,click_mode
    test (12 columns):0,1,..11
    """

    for input_file in [
                  './output/stacking/L_500000_191_0.68164_0422_0730.h5',
                  './output/stacking/L_500000_191_0.68142_0434_0730.h5',
                  './output/stacking/L_0.68018_0914_1667.h5',]:
        score, best_para  = find_best_para(input_file)
        logger.info(f'{input_file},score:{score:0.5f}, best_para:{ best_para }')

        sub_file  = f'./output/sub/st_adj_{score:0.5f}.csv'
        gen_sub_file(input_file, best_para, sub_file)



"""


nohup python -u  core/adj_search.py   > adj_search.log 2>&1 &
"""
