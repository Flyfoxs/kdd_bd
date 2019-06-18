import warnings

from sklearn.metrics import f1_score
from  core.feature import get_query, get_feature_core
warnings.filterwarnings("ignore")

from file_cache.utils.util_pandas import *

begin = '2018-11-03' # '2018-11-17'
end = '2018-12-01'
train_cnt = 0
train_cnt_list = []


@timed()
def adjust_res_ratio( p, adj,):
    adj = adj.copy()
    for i in range(12):
        adj.iloc[:, i] = adj.iloc[:, i] * p[i]

    adj['recommend_mode'] = adj.iloc[:, :12].idxmax(axis=1)
    score_after = f1_score(adj.click_mode.astype(int), adj.recommend_mode.astype(int), average='weighted')
    return -score_after


@timed()
def find_best_para(file):
    para_list = []
    all_click_mode = pd.Series()
    all_recom_old = pd.Series()
    all_recom = pd.Series()

    train = pd.read_hdf(file, 'train')
    old_len = len(train)
    for city in range(4):
        adj = filter_by_data(train.copy(), city)
        logger.info(f'only keep {len(adj)} records from {old_len} records for city#{city}')

        train_cnt_list.append(len(adj))
        raw_score = f1_score(adj.click_mode.values.astype(int), adj.iloc[:, :12].idxmax(axis=1).astype(int), average='weighted')
        all_recom_old = all_recom_old.append(adj.iloc[:, :12].idxmax(axis=1).astype(int))

        from functools import partial
        loss_partial = partial(adjust_res_ratio, adj=adj.copy() )

        initial_coef = [1.12, 0.9, 0.8, 2.43, 2.9, 0.8, 1.85, 0.93, 1.52, 0.91, 1.33, 1.08]
        from scipy.optimize import minimize
        #method = 'nelder-mead'
        method = 'Powell'
        coef_ = minimize(loss_partial, initial_coef, method=method)

        print('coef_=', coef_)

        coef_ =  coef_.x


        for i in range(12):
            adj.iloc[:, i] = adj.iloc[:, i] * coef_[i]


        adj_score_city = f1_score(adj.click_mode.values.astype(int), adj.iloc[:, :12].idxmax(axis=1).astype(int), average='weighted')
        all_click_mode = all_click_mode.append(adj.click_mode)
        all_recom = all_recom.append(adj.iloc[:, :12].idxmax(axis=1).astype(int))

        para_list.append(coef_)
        logger.info(f'for city:{city}, {raw_score} => {adj_score_city}, with para:{coef_}')

    raw_score = f1_score(all_click_mode.astype(int), all_recom_old, average='weighted')
    adj_score = f1_score(all_click_mode.astype(int), all_recom, average='weighted')

    logger.debug(f'Final score for {begin} is {raw_score} => {adj_score}')
    global  train_cnt
    train_cnt = sum(train_cnt_list)
    #raw_score, adj_score, best_para
    return raw_score, adj_score, para_list


def gen_sub_file(input_file, paras, adj_score, raw_score):
    sub = pd.read_hdf(input_file, 'test')

    query = get_feature_core()[['city']]
    sub = sub.join(query)


    sub_file = f'./output/sub/ad1_{adj_score:6.5f}_{raw_score:6.5f}_{begin}_{train_cnt}.csv'
    for city in range(4):
        logger.info(f'There are {len(sub.loc[sub.city==city])} record will be adjust ')
        logger.info(f'Adjust with paras: {paras[city]}')
        for i in range(12):
            sub.loc[sub.city==city, str(i)] = sub.loc[sub.city==city, str(i)] * paras[city][i]

    sub['recommend_mode'] = sub.iloc[:, :12].idxmax(axis=1)

    import csv
    #sub.index = pd.Series(sub.index).apply(lambda val: val.split('-')[-1])
    sub[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL)
    logger.info(f'>>>>Sub file save to {sub_file}')


# input_list = [
#     ('./output/stacking/L_500000_268_0.67818_0439_1462.h5', 1),
#
#     ('./output/stacking/L_500000_190_0.67825_0534_1613.h5', 1), ]
#


def filter_by_data(df, city):
    query = get_feature_core()
    query.req_time = pd.to_datetime(query.req_time)#.dt.date
    query = query.loc[(query.city==city) & (query.label=='train') & (query.req_time>=pd.to_datetime(begin))  & (query.req_time<pd.to_datetime(end)) ]
    print(query.shape)
    return df.loc[df.index.isin(query.sid.astype(int))]

if __name__ == '__main__':
    """
    The stack_file require the format as below:
    train:(13 columns): 0,1,..11,click_mode(不是 recomm_mode,而是原始label)
    test: (12 columns): 0,1,..11

    index is sid, and DF is sorted by index asc
    """
    for begin in [
                  '2018-11-10',
                  '2018-11-17',
                  '2018-11-03',
                  '2018-11-24',
                  '2018-11-22', '2018-11-15', '2018-11-08', '2018-11-01'
                  ]:

        for input_file in [
                                './output/stacking/L_True_2000000_649_0.66997_1035_1524.h5',
                                #'./output/stacking/L_2000000_647_0.67005_0951_0951.h5',
                                #'./output/stacking/L_2000000_649_0.67398_1011_1320.h5', #binary
                                #'./output/stacking/L_2000000_536_0.66944_1038_1685.h5',
                                #'./output/stacking/L_2000000_480_0.66449_0629_1672.h5',
                                #'./output/stacking/L_2000000_336_0.65994_1539_2530.h5', #0.69506305
                              # './output/stacking/L_1500000_336_0.65318_1470_2220.h5',
                              # './output/stacking/L_1500000_336_0.65328_1129_2425.h5',

                          ][:1]:

            raw_score, adj_score, best_para = find_best_para(input_file)
            logger.info(
                f'{input_file},raw_score:{raw_score:0.5f},adj_score:{adj_score:0.5f},\n=== best_para:{ best_para }')
            gen_sub_file(input_file, best_para, adj_score, raw_score)
            # break

"""


nohup python -u  core/adj_search_city.py   > adj_search_city.log 2>&1 &
"""
