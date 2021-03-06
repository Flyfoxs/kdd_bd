from datetime import timedelta

from core.feature import *

from core.split import *
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
from sklearn import datasets

from sklearn.metrics import roc_auc_score,accuracy_score
from core.metric import *
import fire


@timed()
def vali_sub(sub):
    feature = get_feature()


    if 'recommend_mode' not in sub.columns:
        sub['recommend_mode'] = sub.iloc[:, :12].astype(float).idxmax(axis=1)
    feature.index = feature.index.astype(int)
    choose = sub.join(feature)

    col_list = [f'o_seq_{i}' for i in range(7)]
    col_list.append('recommend_mode')
    choose = choose[col_list].astype(int)

    def check_error(row):
        # print(row.recommend_mode, row.iloc[ :-1].values)
        if row.o_seq_0 == 0 and row.recommend_mode == 0:
            return 'zero-0'
        elif row.recommend_mode == 0:
            return 'zero-other'
        elif row.recommend_mode > 0 and row.recommend_mode in row.iloc[:-1].values:
            return 'correct'
        elif row.recommend_mode > 0:
            return 'error'
        else:
            return 'other'

    choose['label'] = choose.apply(lambda item: check_error(item), axis=1)
    error = choose.loc[choose.label.isin(['error', 'other'])]
    if len(error) > 0:
        logger.error(f'❌There are {len(error) } predictions is incorrect.\n {error.label.value_counts()}')
    else:
        logger.info(f'✔️No error is found')

    return choose

@timed()
def gen_sub(file):
    # file = './output/res_False_0.6802.csv'
    res = pd.read_csv(file)
    res.sid = res.sid.astype(int)
    res = res.set_index('sid')
    res['recommend_mode'] = res.idxmax(axis=1)

    feature = get_feature()
    zero_sid = feature[(feature.o_seq_0 == 0) & (feature.click_mode == -1)].index.values
    res.loc[zero_sid.astype(int),'recommend_mode'] = 0

    vali_sub(res)

    import csv
    sub_file = file.replace('res', f'sub/{version}_sub')
    res[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL)
    logger.info(f'Sub file save to {sub_file}')


# def get_groups(cut_point = val_cut_point):
#     feature = get_feature()
#     #feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]
#     feature = feature.loc[(feature.click_mode >= 0) & (feature.o_seq_0_ > 0)].reset_index(drop=True)
#     day_ = feature.date
#     day_ = day_ - min(day_)
#     day_ = day_.dt.days
#     #end = max(day_)
#     return [(day_.loc[day_<=cut_point].index.values ,day_.loc[day_>cut_point].index.values) ]


# dic_ = df_analysis['mode'].value_counts(normalize = True)

##df_analysis['mode']是真实的label的情况
#
# def get_weighted_fscore(y_pred, y_true):
#     f_score = 0
#     for i in range(12):
#         yt = y_true == i
#         yp = y_pred == i
#         f_score += dic_[i] * f1_score(y_true=yt, y_pred= yp)
#     print(f_score)


def filter_index(trn_idx, city=2):
    query = get_feature_core()[['city','sid']].copy()
    query =query.iloc[trn_idx]
    query['idx'] = trn_idx
    return query.loc[query.city==city].idx.values


@timed()
def train_lgb(train_data, orig_X_test, cv=False, args={}, drop_list=[]):


    num_class = 12

    oof = np.zeros((len(train_data), num_class))
    predictions = np.zeros((len(orig_X_test), num_class))
    # start = time.time()
    feature_importance_df = pd.DataFrame()

    # if cv:
    #     folds = KFold(n_splits=num_fold, shuffle=True, random_state=666)
    #     split_fold = folds.split(X_data.values, y_data.values)
    # else:
    #     folds = manual_split()
    #     split_fold = folds.split(X_data, 60-6)
    folds = manual_split()
    split_fold = folds.split(train_data, cv)

    max_iteration = 0
    min_iteration = 99999

    feature_cnt = ()

    # orig_X_test=orig_X_test.loc[orig_X_test.city==2]
    for fold_, (trn_idx, val_idx) in enumerate(tqdm(split_fold, 'Kfold')):
        import gc
        gc.collect()
        with timed_bolck(f'Fold#{fold_}'):
            # val_idx = filter_index(val_idx)
            #print(train_data.shape,trn_idx.shape, val_idx.shape , X_test.shape,trn_idx.max(), val_idx.max() )
            train_x, train_y, val_x, val_y, X_test = extend_split_feature(train_data, trn_idx, val_idx, orig_X_test, drop_list)
            feature_cnt = train_data.shape[0], train_x.shape[1]
            logger.info(f"fold n°{fold_} BEGIN, cv:{cv},all_train:{feature_cnt}, train:{train_x.shape}, val:{val_x.shape}, test:{X_test.shape}, cat:{cate_cols} " )
            trn_data = lgb.Dataset(train_x, train_y, categorical_feature=cate_cols)
            val_data = lgb.Dataset(val_x, val_y , categorical_feature=cate_cols, reference=trn_data)

            # np.random.seed(666)
            """
            boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
        max_depth=-1, n_estimators=3000, objective='multiclass',num_classes=12,
        subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
        learning_rate=0.05, random_state=2019 + index, n_jobs=40, metric="None", importance_type='gain'
        """
            params = {
                # 'nthread': -1,
                # 'verbose':-1,
                # 'num_leaves': 128,
                #### 'min_data_in_leaf': 90,
                # 'feature_fraction':0.5,
                # 'lambda_l1': 0.1,
                # 'lambda_l2': 10,
                #  'max_depth': 6,
                #
                # 'learning_rate': 0.1,
                # 'bagging_fraction': 0.7,

                'objective': 'multiclass',
                'metric': 'None',
                'num_class': num_class,
                #'random_state': 2019,
                # 'device':'gpu',
                # 'gpu_platform_id': 1, 'gpu_device_id': 0
            }
            params = dict(params, **args)

            logger.info(params)

            num_round = 30000
            #num_round = 10
            verbose_eval = 50
            clf = None
            with timed_bolck(f'Train#{fold_}'):
                clf = lgb.train(params,
                                trn_data,
                                num_round,
                                valid_sets=[trn_data, val_data],
                                feval=lgb_f1_score,
                                verbose_eval=verbose_eval,
                                early_stopping_rounds=400)

            max_iteration = max(max_iteration, clf.best_iteration)
            min_iteration = min(min_iteration, clf.best_iteration)

            oof[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration)

            dic_ = val_y.value_counts(normalize=True)
            get_weighted_fscore(val_y.values, oof[val_idx].argmax(axis=1), dic_)
            score = f1_score(val_y.values, oof[val_idx].argmax(axis=1), average='weighted')

            logger.info(f'fold n{fold_} END, cv:{cv}, local_score:{score:6.4f},best_iter:{clf.best_iteration}, val shape:{val_x.shape}')

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = X_test.columns
            fold_importance_df["importance"] = clf.feature_importance()
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            if cv:
                predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
            elif len(args)==0: #not Search model

                all_data, y_data, _, _ , X_test = extend_split_feature(train_data, range(len(train_data)),[], orig_X_test, drop_list)
                all_train = lgb.Dataset(all_data, y_data, categorical_feature=cate_cols)
                logger.info(f'CV is disable, train with full train data with iter:{clf.best_iteration}')
                clf = lgb.train(params,
                                all_train,
                                # num_round,
                                num_boost_round=clf.best_iteration,
                                valid_sets=[all_train],
                                feval=lgb_f1_score,
                                verbose_eval=verbose_eval * 2,
                                )
                predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
                break
            else:
                break
    predictions = predictions / (fold_ + 1)
    if cv:
        score = f1_score(train_data.click_mode.values, oof.argmax(axis=1), average='weighted')

    logger.info(f'cv:{cv}, the final local_score:{score:7.6f}, predictions:{predictions.shape}, params:{params}')
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=[str(i) for i in range(12)])
    predictions.index.name = 'sid'
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    #if cv:
    oof = pd.DataFrame(oof, index=train_data.index, columns=[str(i) for i in range(12)])
    save_stack_feature(oof, predictions, f'./output/stacking/{version}_{cv}_{"_".join(map(str, feature_cnt))}_{score:0.5f}_{min_iteration:04}_{max_iteration:04}.h5')
    return predictions, score, feature_importance_df, f'{min_iteration}_{max_iteration}'

def save_stack_feature(train:pd.DataFrame, test:pd.DataFrame, file_path):
    train_label = train.copy()
    train_label = train_label.loc[train_label.sum(axis=1)>0]
    feature = get_feature()
    train_label['click_mode'] = feature.loc[train_label.index.values, 'click_mode']
    train_label.to_hdf(file_path,'train',mode='a')
    test.to_hdf(file_path, 'test', mode='a')
    logger.info(f'OOF file save to :{file_path}')


def plot_import(feature_importance):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(16, 12));
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
    plt.title('LGB Features (avg over folds)')


def get_search_space():
    space = {"num_leaves":hp.choice("num_leaves", range(60, 100, 10)),
             #"max_depth": hp.choice("max_depth", [7]),
             'lambda_l1': hp.choice("lambda_l1", range(10, 100, 10)),
             'lambda_l2': hp.choice("lambda_l2", range(5,50,5)),
             'feature_fraction': hp.choice("feature_fraction", [0.55, 0.6,0.65, 0.7]),

             'bagging_fraction': hp.choice("bagging_fraction", [0.55, 0.6, 0.65, 0.7]),
             'min_data_in_leaf': hp.choice("min_data_in_leaf", range(70, 160, 10)),

             # 'list_type': hp.choice("list_type", range(0, 10)),
             }
    return space

def search_manual(version='0.678278'):
    """
    version, drop_columns, feature_nums, start_time, end_time, cost, score, server
    :param version:
    :return:
    """
    import pandas as pd
    from tqdm import tqdm
    tmp = ['4_eta_max_p','1_distance_max_p','5_eta','5_distance_max_p','d_hash_6','o_d_hash_6','2_distance_max_p','10_distance','6_eta','7_distance','10_distance_max_p']
    for drop_columns in tqdm(tmp, 'drop_col_list'):
        from core.db_lock import log_begin, log_end

        if log_begin(version, drop_columns):

            res = train_ex(drop_list=drop_columns)

            feature_nums = res['feature_nums']

            score = -res['loss']

            log_end(version, drop_columns, feature_nums, score)
            logger.info(f'Search result:{score:0.5f},{version}, {drop_columns}')
        else:
            logger.warning(f'Can not get lock for {version}, {drop_columns}')


def search_droplist(version='0.678278'):
    """
    version, drop_columns, feature_nums, start_time, end_time, cost, score, server
    :param version:
    :return:
    """
    import pandas as pd
    from tqdm import tqdm
    imp_file = './output/fi_True_579_1215_500000_191_0.6781_.h5'
    tmp = pd.read_hdf(imp_file)
    tmp = tmp.groupby('feature').importance.sum().sort_values(ascending=False).index
    tmp = [item for item in tmp if item not in good_col]
    for drop_columns in tqdm(tmp, 'drop_col_list'):
        from core.db_lock import log_begin, log_end

        if log_begin(version, drop_columns):

            res = train_ex(drop_list=drop_columns)

            feature_nums = res['feature_nums']

            score = -res['loss']

            log_end(version, drop_columns, feature_nums, score)
            logger.info(f'Search result:{score:0.5f},{version}, {drop_columns}')
        else:
            logger.warning(f'Can not get lock for {version}, {drop_columns}')


@timed()
def train_ex(args={}, drop_list='' ):
    drop_list = drop_list.split(',')


    train_data, X_test = get_train_test()
    #train_data = train_data.sample(frac=0.1, random_state=2019 )

    for cv in [True]:
        res, score, feature_importance, best_iteration = train_lgb(train_data, X_test, cv=cv, args=args, drop_list=drop_list )
        #logger.info(f'score:{score:0.6f}, drop_col:{",".join(drop_list)}')
        feature_nums = len(feature_importance.feature.value_counts())
        if len(args) == 0 or cv == True:
            file = f'./output/res_enhance_{cv}_{len(train_data)}_{feature_nums}_{best_iteration}_{score:8.6f}_{"_".join(drop_list)}.csv'
            res.to_csv(file)
            #gen_sub(file)
        else:
            logger.debug('Search model, do not save file')
        feature_importance.to_hdf(f'./output/fi_{cv}_{best_iteration}_{len(train_data)}_{feature_nums}_{score:6.4f}_{"_".join(drop_list)}.h5',key='key')

    res = { 'loss': -score, 'status': STATUS_OK, 'feature_nums':feature_nums, 'attachments': {"message": f'{args} ', } }
    logger.info(res)
    return res

@timed()
def search_paras():
    trials = Trials()
    space = get_search_space()
    best = fmin(train_ex, space, algo=tpe.suggest, max_evals=15, trials=trials)

    logger.debug(f"Best: {best}")
    att_message = [trials.trial_attachments(trial)['message'] for trial in trials.trials]
    for score, para, misc in zip(trials.losses(),
                                 att_message,
                                 [item.get('misc').get('vals') for item in trials.trials]
                                 ):
        logger.debug(f'score:{"%9.6f"%score}, para:{para}, misc:{misc}')


def adjust_res(adj, p):
    p = [round(item, 3) for item in p]
    adj_tmp = adj.copy()

    adj_tmp.loc[adj['3'] > p[0], '3'] = 100
    adj_tmp.loc[adj['6'] > p[1], '6'] = 99
    adj_tmp.loc[adj['4'] > p[2], '4'] = 98
    adj_tmp.loc[adj['8'] > p[3], '8'] = 97
    adj_tmp.loc[adj['0'] > p[4], '0'] = 96

    adj_tmp['recommend_mode'] = adj_tmp.iloc[:, :12].idxmax(axis=1)

    return adj_tmp


if __name__ == '__main__':

    fire.Fire()
    # train_ex()
    # search()
    #gen_sub('output/res_geo_True_500000_191_760_1012_0.6779_sphere_dis.csv')


""""
nohup python -u  core/train.py train_ex > train_profile_lda.log 2>&1 &

nohup python -u  core/train.py train_ex > train_geo_o.log 2>&1 &

nohup python -u  core/train.py train_ex > train_statistics_2.log 2>&1 &

nohup python -u  core/train.py search > search_logloss.log  2>&1 &

nohup python -u  core/train.py search > search_cv_sk.log  2>&1 &

nohup python -u  core/train.py train_ex > train_price_eta.log 2>&1 &

nohup python -u  core/train.py train_ex > train_price_eta_with_zero.log 2>&1 &

nohup python -u  core/train.py train_ex > del.log 2>&1 &

nohup python -u  core/train.py train_ex > 2019_tain_base_on_all.log 2>&1 &

#nohup python -u  core/train.py train_ex > 2019_base_0.69366536.log 2>&1 &

nohup python -u  core/train.py train_ex > base_02_disable_phase1.log 2>&1 &

nohup python -u  core/train.py train_ex > base_13_remove_analysis_deep.log 2>&1 &

nohup python -u  core/train.py train_ex > base_17_3_st.log 2>&1 &

nohup python -u  core/train.py train_ex > del.log 2>&1 &

nohup python -u  core/train.py train_ex > base_17_val_1.log 2>&1 &

nohup python -u  core/train.py train_ex  {} 4_eta_max_p,1_distance_max_p,d_hash_6 > drop_test.log 2>&1 &


nohup python -u  core/train.py search_droplist > search_droplist_3.log 2>&1 &

nohup python -u  core/train.py search_manual > search_manual.log 2>&1 &


nohup python -u  core/train.py train_ex > base_18_none_5.log 2>&1 &
nohup python -u  core/train.py train_ex > base_18_stand_7.log 2>&1 &
nohup python -u  core/train.py train_ex > base_18_minmax_6.log 2>&1 &

nohup python -u  core/train.py train_ex > base_24.log 2>&1 &


nohup python -u  core/train.py train_ex > base_20_all_sz.log 2>&1 &
"""