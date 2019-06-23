import json
import warnings
from functools import lru_cache



from deprecated import deprecated
from file_cache.utils.util_log import timed_bolck
from file_cache.cache import file_cache
from file_cache.utils.reduce_mem import *
from file_cache.utils.util_pandas import *
from sklearn.decomposition import TruncatedSVD

from core.config import *

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



@file_cache()
def get_plan_model_sequence():
    deep_plan = get_plan_original_deep() #get_plan_model_sequence
    deep_plan = deep_plan.set_index('sid')
    res_list = []
    #res_list.append(deep_plan.dummy_sid)
    for i in range(12):
        recomm_mode = deep_plan.loc[deep_plan.sn==i, 'transport_mode' ].to_frame()
        #print(type(recomm_mode))
        recomm_mode.rename({'transport_mode':f'o_seq_{i}'}, inplace = True, axis=1)
        if recomm_mode.iloc[:,-1].sum()==0 :
            logger.info(f'No sid have more than {i} plan')
            break
        logger.info(f'recomm_mode shape:{recomm_mode.shape}')
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
    sigle_model = sigle_model.drop_duplicates('sid')

    logger.warning(f'There are {old_len - len(sigle_model)} records were remove from {old_len} for mode:{model}')

    #sigle_model['price'] = sigle_model['price'].fillna(0)  # .astype(float)

    #sigle_model = sigle_model.fillna(0).astype(int)#.set_index('dummy_sid')

    mini_plan = sigle_model.loc[:, plan_items + plan_rank]


    #TODO
    mini_plan['pe_eta_price'] = mini_plan['eta']/mini_plan['price']
    mini_plan['pe_dis_price'] = mini_plan['distance']/mini_plan['price']
    mini_plan['pe_dis_eta']   = mini_plan['distance']/mini_plan['eta']

    mini_plan['pe_price_eta'] = mini_plan['price']/mini_plan['eta']
    mini_plan['pe_price_dis'] = mini_plan['price']/mini_plan['distance']
    mini_plan['pe_eta_dis']   = mini_plan['eta']/mini_plan['distance']

    #mini_plan['transport_mode'] = mini_plan['transport_mode']//model

    mini_plan['sid']       = sigle_model.sid
    #mini_plan['dummy_sid'] = sigle_model.dummy_sid

    mini_plan = mini_plan.set_index('sid')

    return mini_plan.add_prefix(f'{model:02}_')

@timed()
# @lru_cache()
def get_original(file):
    df =  pd.read_csv(f'{input_folder}/{file}', dtype=type_dict)
    return df


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
        select_col = [ f'{i:02}_{item}' for i in range(12)]
        col_list = [col for col in plan.columns if col in select_col]
        plan_percent = plan.loc[:, col_list].copy()
        total = plan_percent.max(axis=1)
        for col in plan_percent:
            plan_percent[f'{col}_max_p'] = round(plan_percent[col] / total, 4)
            del plan_percent[col]

        res_list.append(plan_percent)
    res = pd.concat(res_list, axis=1)

    col_list = ['01_distance_max_p', '01_eta_max_p', '01_price_max_p', '10_distance_max_p', '10_eta_max_p', '10_price_max_p', '11_distance_max_p', '11_eta_max_p', '11_price_max_p', '02_distance_max_p', '02_eta_max_p', '02_price_max_p', '03_distance_max_p', '03_eta_max_p', '03_price_max_p', '04_distance_max_p', '04_eta_max_p', '04_price_max_p', '05_distance_max_p', '05_eta_max_p', '05_price_max_p', '06_distance_max_p', '06_eta_max_p', '06_price_max_p', '07_distance_max_p', '07_eta_max_p', '07_price_max_p', '08_distance_max_p', '08_eta_max_p', '08_price_max_p', '09_distance_max_p', '09_eta_max_p', '09_price_max_p', ]
    # res.columns.set_levels([ f'{item[1]}_p' for item in res.columns ],level=1,inplace=True)
    # res.columns = [ (item[0], f'{item[1]}_p') for item in res.columns]
    res = res.loc[:, col_list]
    res = res.sort_index(axis=1, level=1)
    return res


@file_cache()
def get_plan_nlp():

    N_COM = 10

    from sklearn.feature_extraction.text import TfidfVectorizer

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
@reduce_mem()
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

    plan['sid'] = plan.index

    plan.index.name = 'index'

    return plan


@deprecated
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

@timed()
@file_cache()
def get_plan_original_deep():
    plan_list = []
    for plan_file, phase in [('train_plans_phase1.csv', 1), ('train_plans_phase2.csv', 2), ('test_plans.csv', 2)]:
        original_plan = get_original(plan_file)
        for index_sn, row in original_plan.iterrows():
            plans = json.loads(row.plans)
            plan_sn = 0
            mode_list = []
            for single_plan in plans:
                cur_mode = single_plan['transport_mode']
                if True or cur_mode not in mode_list:
                    mode_list.append(cur_mode)
                    single_plan['sn'] = plan_sn
                    plan_sn += 1
                    single_plan['sid'] = row.sid
                    single_plan['phase'] = phase
                    plan_list.append(single_plan)
                # else:
                #     logger.info(f'Already have moede:{cur_mode} for sid:{row.sid}')
    res = pd.DataFrame(plan_list).astype(int)

    logger.debug(f'Try to fillna for plan info:{len(res)}')
    res[plan_items] = res.loc[:,plan_items].fillna('0').replace({'':'0'}).astype(int)

    res['price_rank'] = res[['sid', 'price']].groupby(['sid'])['price'].rank(method='min')
    res['distance_rank'] = res[['sid', 'distance']].groupby(['sid'])['distance'].rank(method='min')
    res['eta_rank'] = res[['sid', 'eta']].groupby(['sid'])['eta'].rank(method='min')

    return res




#@lru_cache()
@file_cache()
@reduce_mem()
def get_plan_original_wide():
    res_list = []
    base_list = []
    for file, phase in [('train_plans_phase1.csv', 1), ('train_plans_phase2.csv', 2), ('test_plans.csv', 2)]:
        base = get_original(file)
        base['phase'] = phase
        base_list.append(base)
    base = pd.concat(base_list)
    from multiprocessing import Pool as ThreadPool  # 进程

    #Initial
    get_plan_original_deep()

    #get_plan_mini_ex = partial(get_plan_mini)

    pool = ThreadPool(6)
    plan_list = pool.map(get_plan_mini, tqdm(range(1, 12)), chunksize=1,)


    plan_part = pd.concat(plan_list, axis=1)

    plan_part.index.name = 'sid'

    plan_part = plan_part.reset_index()

    plan_part = pd.merge(base, plan_part, on='sid')
    return plan_part.set_index('sid')

@timed()
# @lru_cache()
@file_cache()
def get_query():
    import geohash as geo
    train_1 = get_original('train_queries_phase1.csv')
    train_1['label'] = 'train'
    train_1['phase'] = 1

    train_2 = get_original('train_queries_phase2.csv')
    train_2['label'] = 'train'
    train_2['phase'] = 2

    test = get_original('test_queries.csv')
    test['label'] = 'test'
    test['phase'] = 2

    train_query = pd.concat([train_1, train_2, test])



    train_query.pid = train_query.pid.fillna(0)
    train_query.pid = train_query.pid.astype(int)
    train_query.req_time = pd.to_datetime(train_query.req_time)
    train_query['date'] = train_query.req_time.dt.date

    day_ = pd.to_datetime(train_query.req_time).dt.date
    day_ = day_ - min(day_)
    day_ = day_.dt.days

    with timed_bolck('request_time_process'):
        train_query['day']  = day_
        train_query['weekday'] = train_query.req_time.dt.weekday
        train_query['hour'] = train_query.req_time.dt.hour
        train_query['weekend'] = train_query.weekday // 5

    with timed_bolck('o_d_address_process'):
        train_query['o0'] = train_query.o.apply(lambda item: item.split(',')[0]).astype(float)
        train_query['o1'] = train_query.o.apply(lambda item: item.split(',')[1]).astype(float)

        train_query['d0'] = train_query.d.apply(lambda item: item.split(',')[0]).astype(float)
        train_query['d1'] = train_query.d.apply(lambda item: item.split(',')[1]).astype(float)

        train_query['sphere_dis'] = train_query.apply(lambda row: getDistance(row.o0,row.o1, row.d0,row.d1,), axis=1)

    #train_query['city'] =  train_query.apply(lambda val: get_city(val.o0, val.o1), axis=1)

    with timed_bolck('hash_geo_convert'):
        for precision in [5,6]:
            train_query[f'o_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.o1, row.o0, precision=precision),
                                                                   axis=1)

            train_query[f'd_hash_{precision}'] = train_query.apply(lambda row: geo.encode(row.d1, row.d0, precision=precision),
                                                                   axis=1)

            train_query[f'o_d_hash_{precision}'] = train_query[f'o_hash_{precision}'] + '_' + train_query[f'd_hash_{precision}']

    with timed_bolck('request_join_click'):
        click = get_click()
        train_query = pd.merge(train_query, click, how='left', on='sid')
        train_query.loc[train_query.label=='test', 'click_mode'] = -1
        train_query.click_mode = train_query.click_mode.fillna(0).astype(int)
        del train_query['click_time']

    train_query.index = train_query.sid
    train_query.index.name = 'index'
    return train_query.sort_index()

@timed()
def get_plan_stati_feature_sid():
    res_list = []

    plans = get_plan_original_deep() #get_plan_stati_feature_sid
    plans = plans.drop_duplicates(['sid', 'transport_mode'])

    for col in plan_items_mini:
        tmp = plans.groupby('sid')[col].agg(['min', 'max', 'mean', 'std']).add_prefix(f'ps_{col}_')
        res_list.append(tmp)

    tmp = plans.groupby('sid')['transport_mode'].agg(['count','nunique']).add_prefix(f'ps_transport_mode_')
    tmp['ps_transport_mode_nunique'] = tmp['ps_transport_mode_count'] - tmp['ps_transport_mode_nunique']
    res_list.append(tmp)
    return pd.concat(res_list, axis=1)

def get_mode(ser):
    return ser.value_counts().index[0]

def get_mode_count(ser):
    return ser.value_counts().values[0]

@timed()
def get_stati_feature_pid():

    res_list = []

    query = get_query()
    query_mini = query.loc[:, ['pid', 'sid']]

    plans = get_plan_original_deep() #get_stati_feature_pid
    plans = plans.drop_duplicates(['sid', 'transport_mode'])
    plans = pd.merge(plans, query_mini, how='left', on='sid')

    pid_mode = plans.groupby('pid').transport_mode.agg(
        ['median', 'std', 'nunique', 'count', get_mode, get_mode_count]).add_prefix('s_pid_m_')
    res_list.append(pid_mode)

    # query = get_query()
    # query.head()

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
        tmp = query.groupby(gp_level)[f'{i:02}_transport_mode'].agg({f'sugg_{direct}_{i:02}': 'sum'})
        tmp[f'sugg_{direct}_{i:02}'] = tmp[f'sugg_{direct}_{i:02}'] // i

        res_list.append(tmp.astype(int))

    tmp = query.groupby(gp_level)['day'].agg(
        {f'day_appear_nunique_{direct}': 'nunique', f'count_appear_{direct}': 'count'})
    res_list.append(tmp.astype(int))

    tmp = pd.concat(res_list, axis=1)  # .sort_values('count_appear', ascending=False).loc['wx4g0w'].sort_index()

    for i in range(1, 12):
        tmp[f'sugg_{direct}_{i:02}_per'] = tmp[f'sugg_{direct}_{i:02}'] / tmp[f'count_appear_{direct}']

    tmp = tmp.add_prefix(prefix)
    # tmp.index.name = f'{direct}_hash_{hash_precision}'
    res = tmp.reset_index()

    return res


# @lru_cache()
def get_click():
    click_1 =  get_original('train_clicks_phase1.csv')
    click_2 =  get_original('train_clicks_phase2.csv')

    return pd.concat([click_1, click_2])

def get_profile(n_topics = 5):
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

    lda = get_profile_lda(n_topics)

    return pd.concat([profile_data,lda], axis=1)


@file_cache()
def get_profile_lda(n_topics):
    def get_profile_text():
        profile = get_original('profiles.csv').astype(int)
        profile = profile.set_index('pid')
        col_list = profile.columns
        for col in tqdm(col_list):
            profile[col] = profile[col].map({0: ' ', 1: col})

        res = profile.apply(lambda row: ' '.join(row), axis=1)
        return res

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    profiles = get_profile_text()
    vectorizer = CountVectorizer()
    #transformer = TfidfTransformer()
    cntTf = vectorizer.fit_transform(profiles)


    lda = LatentDirichletAllocation(n_components=n_topics,
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

@deprecated(reason='useless')
def extend_c2v_feature(c_list=['weekday' , 'hour']):
    feature = get_feature().reset_index()
    original_c2v = pd.read_csv('./output/c2v.txt', delimiter=' ', skiprows=1, header=None)

    for col in c_list:
        c2v = original_c2v.copy().add_prefix(f'{col}_')
        c2v = c2v.rename({f'{col}_0': col}, axis=1)
        feature[col] = feature[col].apply(lambda item: f'{col}={item}')
        feature = pd.merge(feature,c2v, how='left', on=col)
        del feature[col]
    feature = feature.fillna(0)

    return feature.set_index('sid')

@timed()
def get_feature_ex_bin():
    feature = get_feature().copy()
    feature['sid']  = feature.index
    #feature.index = pd.Series(feature.index).apply(lambda val: val.split('-')[1]).astype(int)

    bin_feature = pd.read_csv('./input/tmp/oof_train_test.csv', index_col='sid')

    feature = pd.concat([feature, bin_feature], axis=1)

    feature = feature.set_index('sid')

    return feature


@timed()
def get_train_test():
    #feature = get_feature_ex_bin()#.copy()
    feature = get_feature().copy()

    #feature = feature.sample(frac=0.2)

    #feature = feature.sort_values(['phase','sid'])

    # feature['sid'] = feature.index
    # tmp = get_plan_analysis_deep()
    # feature = pd.merge(feature, tmp, left_on='o_seq_0', right_on='transport_mode')
    # feature = feature.set_index('sid')

    if disable_phase1 :
        old_len = len(feature)
        feature = feature.loc[feature.phase==2]
        logger.info(f'Remove {old_len-len(feature)} phase#1 data from {old_len} rows:disable_phase1#{disable_phase1} ')

    #feature = extend_c2v_feature().copy()

    click_mode = feature.click_mode
    del feature['click_mode']
    feature['click_mode'] = click_mode.astype(int)

    #logger.info(check_exception(feature).head(4))

    for precision in [6]:
        feature[f'o_d_hash_{precision}'] = feature[f'o_d_hash_{precision}'].astype('category').cat.codes.astype(int)
        feature[f'd_hash_{precision}'] = feature[f'd_hash_{precision}'].astype('category').cat.codes.astype(int)
        feature[f'o_hash_{precision}'] = feature[f'o_hash_{precision}'].astype('category').cat.codes.astype(int)

    train_data = feature.loc[(feature.click_mode >= 0)]

    X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

    logger.info((len(train_data), len(X_test.columns) , list(X_test.columns)))

    return train_data, X_test

@timed()
# @lru_cache()
@file_cache()
@reduce_mem()
def get_feature_core():
    query = get_query()

    query['city'] =  get_city_fea()

    plans = get_plans()
    del plans['phase']


    #plan_cat = get_plan_cat()
    #plans.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in plans.columns]



    #del plans['plan_time']

    #logger.info(query.columns)
    #logger.info( list(plans.columns) )
    with timed_bolck('Merge query plan'):
        query = pd.merge(query, plans, how='left', on='sid')

        time_gap = (pd.to_datetime(query.plan_time) - pd.to_datetime(query.req_time)).dt.total_seconds()
        query['time_gap'] = time_gap.where(time_gap >= 0, -1)

        #Fix the distance precision issue
        query.loc[query.sphere_dis <= 10, 'sphere_dis' ] = query.loc[query.sphere_dis <= 10, 'ps_distance_min']

    with timed_bolck('Merge geo_percentage'):
        for direct in ['o']:
            precision = 6
            gp_level = [f'{direct}_hash_{precision}']
            geo_hash = get_geo_percentage(query, direct, gp_level, )
            query = pd.merge(query, geo_hash, how='left', on=gp_level)

    # for i in range(1, 12):
    #     query[f'{i}_distance_ratio'] = query[f'{i}_distance']/query['raw_dis']

    #query = pd.merge(query, plan_cat, how='left', on='sid')

    #if group is not None and 'profile' in group:
    with timed_bolck('Merge profile'):
        profile = get_profile(5)
        query = pd.merge(query, profile, how='left', on='pid')

    with timed_bolck('Merge static feature'):
    #statistics, information
        stat = get_stati_feature_pid()
        query = pd.merge(query, stat, how='left', on='pid')
        query.pid = query.pid.astype(int)
        logger.info('Finish merge stat feature')

    with timed_bolck('fillna'):
        query = query.fillna(0)
        logger.info('Finish fillna')

    with timed_bolck('move click_mode to end'):
        click_mode = query.click_mode
        del query['click_mode']
        query['click_mode'] = click_mode

    query.index = query.sid.astype(int)

    return query

# @lru_cache()
def get_city_fea():
    core = get_query()
    return core.apply(lambda val: get_city(val.o0, val.o1), axis=1)

def get_city(x, y):
    d1 = (x-116.41) ** 2 + (y-39.91) ** 2
    d2 = (x-121.43) ** 2 + (y-31.20) ** 2
    d3 = (x-114.06) ** 2 + (y-22.54) ** 2
    d4 = (x-113.26) ** 2 + (y-23.13) ** 2
    distance = [d1,d2,d3,d4]
    return np.argmin(distance)


@timed()
@file_cache()
def get_direction():
    query = get_query()
    def bearing_array(lat1, lng1, lat2, lng2):
        #AVG_EARTH_RADIUS = 6378.137  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    return query.apply(lambda row: bearing_array(row.o0,row.o1, row.d0,row.d1,), axis=1)

@timed()
# @lru_cache()
@file_cache()
@reduce_mem()
def get_feature():

    query = get_feature_core()

    if 'plans' in query:
        del query['plans']

    #query['city'] = get_city_fea()

    query['o_d_pid'] = get_o_d_pid()
    # query['direction'] = get_direction()

    # tmp = pd.read_csv('./input/tmp/phase_2_node2vec.csv')
    # tmp.rename(columns={'node_od': 'o'}, inplace=True)
    # query = query.merge(tmp, how='left', on='o')
    # #feature.shape

    triple_gp = get_triple_gp()
    query[triple_gp.columns] = triple_gp

    bin_gp = get_bin_gp()
    query[bin_gp.columns] = bin_gp

    cv_feature = get_cv_feature()
    query[cv_feature.columns] = cv_feature

    #Make click mode the last col
    click_mode = query.click_mode
    del query['click_mode']
    query['click_mode'] = click_mode

    logger.info('Finish merge click feature')
    query.index = query.sid.astype(int)
    logger.info('Finish set index')


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
        return round(distance)  # meter
    except:
        return 1



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

# @lru_cache()
def get_resample_sid():
    st = pd.read_hdf('./output/stacking/L_500000_334_0.67787_0845_1443.h5', 'train')
    st['predict'] = st.iloc[:, :12].idxmax(axis=1)
    st['correct'] =  np.abs(st.predict.astype(int) - st.click_mode.astype(int))

    error = st.loc[st.correct>0]
    error['resample'] = False
    for label in range(12):
        if label not in [0, 3, 4, 6, 9]:
            tmp = error.loc[error.click_mode==label].sort_values(str(label), ascending=False)
            cnt = len(tmp)//20
            print(cnt)
            error.loc[ tmp.iloc[:cnt].index, 'resample' ] =True


    return error.loc[error['resample']==True].reset_index().sid.values


def sample_ex(df):

    print(df.click_mode.value_counts())
    resample_list = get_resample_sid()
    double = df.loc[df.index.isin(resample_list)]
    logger.info(f'There are {len(double)} records are resample from {len(df)} records')

    df = df.append(double)
    print(df.click_mode.value_counts())
    #check_exception((df))
    return df

@timed()
def extend_split_feature(df, trn_idx, val_idx ,  X_test, drop_list):
    val = df.iloc[val_idx].copy()

    val_x = val.iloc[:, :-1]
    val_y = val.click_mode

    logger.info(f'There are {len(val_x.day.value_counts())} days, {len(val_x)} rows records for validate')

    train = df.iloc[trn_idx].copy()

    # old_len = len(train)
    # resample_list = get_resample_sid()
    # train = train.loc[~train.index.isin(resample_list)]
    # logger.info(f'There are {old_len - len(train)} records are remove from df#{old_len}')
    #
    # if mode_list:
    #     train = train.loc[train.click_mode.isin(mode_list)]

    click_mode = train.click_mode.astype(int)
    del train['click_mode']
    train['click_mode'] = click_mode


    train_x = train.iloc[:, :-1]
    train_y = train.click_mode

    train_x = remove_col(train_x, drop_list).fillna(0)
    val_x   = remove_col(val_x, drop_list).fillna(0)
    X_test = remove_col(X_test, drop_list).fillna(0)



    for col, type_ in val_x.dtypes.sort_values().iteritems():
        if type_ not in ['float16', 'float32', 'int8', 'int64', 'int16', 'int32', 'float64']:
            logger.error((col, type_))


    logger.info(f'====extend_split_feature Train:{train_x.shape}, val:{val_x.shape}, col_list:{list(val_x.columns)}')
    #Drop end

    return train_x, train_y, val_x, val_y, X_test



# @lru_cache()
def get_drop_list_std(thres_hold=0):
    feature = get_feature()
    tmp = feature.describe().T
    tmp = tmp.loc[tmp['std'] <= thres_hold].sort_values('std')

    std_drop = list(tmp.index.values)
    logger.info(f'Base on {thres_hold}, get drop list:{len(std_drop)},{std_drop}')
    return std_drop

def remove_col(train, drop_list):
    # Drop begin
    remove_list = ['o_d_hash_5', 'd_hash_5', 'o_hash_5', 'plans',
                   'o', 'd', 'label', 'req_time', 'click_time', 'date',
                   'day', 'plan_time', 'sphere_dis', 'en_label', 'time_gap',
                   'sid', 'phase'
                   #'10_eta',

                   # 's_pid_o_hash_m_per', 's_pid_d_hash_m_per',
                   ]
    remove_list.extend(drop_list)

    #remove_list.extend(get_drop_list_std(0.02))

    remove_list.extend([col for col in train.columns if col.endswith('_rank')])

    remove_list.extend([col for col in train.columns if col.startswith('s_')])

    #remove_list.extend([col for col in train.columns if col.startswith('cv_')])

    # pe_eta_price and so on
    #remove_list.extend([col for col in train.columns if '_pe_' in col])
    #remove_list.extend([col for col in train.columns if 'ps_' in col])
    # remove_list.extend([col for col in [ f'{i}_transport_mode' for i in range(1, 12)]])
    logger.info(f'Final remove list:{remove_list}')
    train = train.drop(remove_list, axis=1, errors='ignore')
    return train#.loc[:, [item for item in col_order if item in train.columns ] ]


from core.split import *
@timed()
@file_cache()
@reduce_mem()
def cv_feat(cols):
    def label_mean(data, feat_set, cols):
        res = np.zeros((data.shape[0], 12))
        for click_mode in range(12):
            feat_set['cur_mode'] = feat_set.click_mode == click_mode
            cols_label = feat_set.groupby([cols], as_index=False)['cur_mode'].agg({'feats': 'mean'})
            res[:, click_mode] = data[[cols]].merge(cols_label, 'left', [cols])['feats'].fillna(0).values
        return pd.DataFrame(res, index=data.index).add_prefix(f'cv_{cols}_')

    feature = get_feature_core()
    feat_set = feature.loc[feature.label == 'train']
    test_set = feature.loc[feature.label == 'test']

    #     print(feature.shape, feat_set.shape, test_set.shape)
    # print(test_set.head(3))

    # result     = pd.DataFrame()
    label_fold = pd.DataFrame()
    res_list = []
    from core.split import manual_split
    kf = manual_split.split_sk(feat_set)
    for k, (train_fold, test_fold) in enumerate(kf):
        result_tmp = label_mean(feat_set.iloc[test_fold, :], feat_set.iloc[train_fold, :], cols)
        res_list.append(result_tmp)
        test_tmp = label_mean(test_set, feat_set.iloc[train_fold, :], cols)

        label_fold = test_tmp + label_fold if not label_fold.empty else test_tmp
    label_fold = label_fold / 5
    #     print(label_fold.sum().sum())
    #     #print(label_fold.head(3))
    res_list.append(label_fold)
    result = pd.concat(res_list)
    print(result.shape, label_fold.shape)
    return result


@timed()
def get_cv_feature():
    core = get_feature_core()
    cv_feature = core.loc[:,['sid']]
    for col in tqdm(['pid', 'o_hash_6',
                'd_hash_6', 'o_d_hash_6',
                'o_seq_0',
                'o_seq_1',
                'o_seq_2',
                'o_seq_3',
                'o_seq_4',
                'o_seq_5',
                'o_seq_6',
                'o_seq_7',
                    # 'city'
                     ]):
        res = cv_feat(col)
        cv_feature[res.columns] = res
        cv_feature.index = cv_feature.sid.astype(int)
    return cv_feature



@timed()
#@file_cache()
@reduce_mem()
def get_triple_gp():
    feature = get_feature_core()

    feature['sphere_dis_bins'] = pd.cut(feature.sphere_dis, bins=20).cat.codes
    to_group = [
        'pid', 'o', 'd', 'o0', 'o1', 'd0', 'd1', 'weekday',
        'weekend', 'hour', 'sphere_dis_bins'
        # 'req_time_dow', 'req_is_weekend', 'sphere_dis_bins', 'odl2_dis_bins',
        # 'Recommand_0_transport_mode','Recommand_1_transport_mode','Recommand_2_transport_mode','price_inMin_0_transport_mode'
    ]

    gen = []
    for i in tqdm(range(len(to_group))):
        for j in range(i + 1, len(to_group)):
            for k in range(j + 1, len(to_group)):
                gp = [to_group[i], to_group[j], to_group[k]]
                if ('_'.join(gp) + '_agg_count' not in feature.columns):
                    gen.append(gp)

    res = feature.loc[:, ['sid']]
    for i in tqdm(gen):
        res['_'.join(i) + '_agg_count'] = feature.groupby(i)['pid'].transform('count')

    res.index = res.sid.astype(int)
    del res['sid']

    # from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # ss = MinMaxScaler()
    # logger.info(f'Scaler with:{type(ss).__name__}')
    # st_res = ss.fit_transform(res)
    # res = pd.DataFrame(np.round(st_res,6), index = res.index, columns=res.columns)

    return res


@timed()
#@file_cache()
@reduce_mem()
def get_bin_gp():
    feature = get_feature_core()

    feature['sphere_dis_bins'] = pd.cut(feature.sphere_dis, bins=20).cat.codes
    to_group = [
        'pid', 'o', 'd', 'o0', 'o1', 'd0', 'd1', 'weekday',
        'weekend', 'hour', 'sphere_dis_bins'
        # 'req_time_dow', 'req_is_weekend', 'sphere_dis_bins', 'odl2_dis_bins',
        # 'Recommand_0_transport_mode','Recommand_1_transport_mode','Recommand_2_transport_mode','price_inMin_0_transport_mode'
    ]

    gen = []
    for i in tqdm(range(len(to_group))):
        for j in range(i + 1, len(to_group)):
            #for k in range(j + 1, len(to_group)):
                gp = [to_group[i], to_group[j]]
                if ('_'.join(gp) + '_agg_count' not in feature.columns):
                    gen.append(gp)

    res = feature.loc[:, ['sid']]
    for i in tqdm(gen):
        res['_'.join(i) + '_agg_count'] = feature.groupby(i)['pid'].transform('count')

    res.index = res.sid.astype(int)
    del res['sid']

    # from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # ss = MinMaxScaler()
    # logger.info(f'Scaler with:{type(ss).__name__}')
    # st_res = ss.fit_transform(res)
    # res = pd.DataFrame(np.round(st_res,6), index = res.index, columns=res.columns)

    return res



@file_cache()
def get_o_d_pid():
    query = get_query()
    with timed_bolck('o_d_pid'):
        tmp = query.groupby(['o', 'd'])['pid'].transform(lambda x: np.mean(x.value_counts()))
        tmp.index = query.sid
    return tmp


@deprecated(reason='loss score')
def get_plan_analysis_deep():
    data = get_plan_original_deep()
    stat_9 = data[['transport_mode']].drop_duplicates()
    for i in tqdm(['transport_mode']):
        tmp = data[[i, 'price', 'eta', 'distance']].groupby([i]).agg(
            {'price': ['sum', 'mean', 'min', 'max', 'std', 'skew', get_mode, get_mode_count],
             'eta': ['sum', 'mean', 'min', 'max', 'std', 'skew', get_mode, get_mode_count],
             'distance': ['sum', 'mean', 'min', 'max', 'std', 'skew', get_mode, get_mode_count]})

        print('=' * 100)
        tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]
        tmp = tmp.add_prefix(i + '_').reset_index()
        stat_9 = stat_9.merge(tmp, how='left', on=i)

    data = get_feature_core()
    for i in tqdm(['o_seq_0']):
        tmp = data[[i, 'sphere_dis']].groupby([i]).agg(
            {'sphere_dis': ['sum', 'mean', 'min', 'max', 'std', 'skew', get_mode, get_mode_count],
             # 'odl2_dis': ['sum', 'mean', 'min', 'max', 'std', 'skew', get_mode, get_mode_count],
             })

        tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]
        tmp.index.name = 'transport_mode'
        tmp = tmp.add_prefix(i + '_').reset_index()
        stat_9 = stat_9.merge(tmp, how='left', on='transport_mode')

    stat_9['is_null_sum'] = stat_9[[i for i in stat_9.columns if i not in ['transport_mode']]].isnull().sum(axis=1)

    return stat_9

if __name__ == '__main__':
    get_query()
    get_direction()
    # get_feature_core()
    # get_triple_gp()
    # get_o_d_pid()
    #
    # get_feature_core()
    # get_feature()
    # get_plan_original_deep() # main
    # get_plan_original_wide()

    """
    nohup python -u  core/feature.py   > feature.log  2>&1 &
    """

