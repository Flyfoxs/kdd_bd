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

    test = feature.loc[(feature.label == 'test') &
                       (feature.click_mode == -1) &
                       (feature.o_seq_0 == 0)]

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


def gen_sub(file):
    # file = './output/res_False_0.6802.csv'
    res = pd.read_csv(file)
    res.sid = res.sid.astype(object)
    res = res.set_index('sid')
    res['recommend_mode'] = res.idxmax(axis=1)

    zero_sid = get_feature()
    zero_sid = zero_sid[(zero_sid.o_seq_0 == 0) & (zero_sid.click_mode == -1)].index.values
    res.loc[zero_sid,'recommend_mode'] = 0

    vali_sub(res)

    import csv
    sub_file = file.replace('res', 'sub/sub')
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


@timed()
def train_lgb(X_data, y_data, X_test, cv=False, args={}):

    num_class = 12

    oof = np.zeros((len(y_data), num_class))
    predictions = np.zeros((len(X_test), num_class))
    # start = time.time()
    feature_importance_df = pd.DataFrame()

    # if cv:
    #     folds = KFold(n_splits=num_fold, shuffle=True, random_state=666)
    #     split_fold = folds.split(X_data.values, y_data.values)
    # else:
    #     folds = manual_split()
    #     split_fold = folds.split(X_data, 60-6)
    folds = manual_split()
    split_fold = folds.split(X_data)

    max_iteration = 0
    min_iteration = 99999

    for fold_, (trn_idx, val_idx) in enumerate(tqdm(split_fold, 'Kfold')):
        logger.info(f"fold n°{fold_} BEGIN, cv:{cv},train:{trn_idx.shape}, val:{val_idx.shape}, test:{X_test.shape}, cat:{cate_cols} " )
        trn_data = lgb.Dataset(X_data.iloc[trn_idx], y_data.iloc[trn_idx], categorical_feature=cate_cols)
        val_data = lgb.Dataset(X_data.iloc[val_idx], y_data.iloc[val_idx], categorical_feature=cate_cols, reference=trn_data)

        # np.random.seed(666)
        params = {
            'verbose':-1,
            'num_leaves': 80,
            'min_data_in_leaf': 100,
            'feature_fraction':0.65,
            'lambda_l1': 20,
            'lambda_l2': 5,
            'max_depth': 6,

            'learning_rate': 0.1,
            'bagging_fraction': 0.7,

            'objective': 'multiclass',
            'metric': 'None',
            'num_class': num_class,
            # 'device':'gpu',
            #'gpu_platform_id': 1, 'gpu_device_id': 0
        }
        params = dict(params, **args)

        logger.info(params)

        num_round = 30000
        verbose_eval = 50
        clf = lgb.train(params,
                        trn_data,
                        num_round,
                        valid_sets=[trn_data, val_data],
                        feval=lgb_f1_score,
                        verbose_eval=verbose_eval,
                        early_stopping_rounds=400)

        max_iteration = max(max_iteration, clf.best_iteration)
        min_iteration = min(min_iteration, clf.best_iteration)

        oof[val_idx] = clf.predict(X_data.iloc[val_idx], num_iteration=clf.best_iteration)

        dic_ = y_data.iloc[val_idx].value_counts(normalize=True)
        get_weighted_fscore(y_data.iloc[val_idx].values, oof[val_idx].argmax(axis=1), dic_)
        score = f1_score(y_data.iloc[val_idx].values, oof[val_idx].argmax(axis=1), average='weighted')

        logger.info(f'fold n{fold_} END, cv:{cv}, local_score:{score:6.4f},best_iter:{clf.best_iteration}, val shape:{X_data.iloc[val_idx].shape}')

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_data.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if cv:
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
        elif len(args)==0: #not Search model
            all_train = lgb.Dataset(X_data, y_data, categorical_feature=cate_cols)
            clf = lgb.train(params,
                            all_train,
                            # num_round,
                            num_boost_round=clf.best_iteration,
                            valid_sets=[all_train],
                            feval=lgb_f1_score,
                            verbose_eval=verbose_eval * 2,
                            )
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
            logger.info(f'CV is disable, will train with full train data with iter:{clf.best_iteration}')
            break
    predictions = predictions / (fold_ + 1)
    if cv:
        score = f1_score(y_data.values, oof.argmax(axis=1), average='weighted')

    logger.info(f'cv:{cv}, the final local_score:{score:6.4f}, predictions:{predictions.shape}, params:{params}')
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=[str(i) for i in range(12)])
    predictions.index.name = 'sid'
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    if cv:
        oof = pd.DataFrame(oof, index=X_data.index, columns=[str(i) for i in range(12)])
        save_stack_feature(oof, predictions, f'./output/stacking/L_{score:0.5f}_{min_iteration:04}_{max_iteration:04}.h5')
    return predictions, score, feature_importance_df, f'{min_iteration}_{max_iteration}'

def save_stack_feature(train:pd.DataFrame, test:pd.DataFrame, file_path):
    train_label = train.copy()
    feature = get_feature()
    train_label['click_mode'] = feature.loc[train_label.index.values, 'click_mode']
    train_label.to_hdf(file_path,'train',mode='a')
    test.to_hdf(file_path, 'test', mode='a')


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


@timed()
def train_ex(args={}):

    for sn, drop_list in enumerate([
        #['date', 'day'],

        #['d_hash_6'],
        ['date'],
        [],
    ]):

        for ratio in range(1):
            train_data, X_test = get_train_test(drop_list)

            X_data, y_data = train_data.iloc[:, :-1], train_data.iloc[:, -1]

            for cv in [True,]:
                res, score, feature_importance, best_iteration = train_lgb(X_data, y_data, X_test, cv=cv, args=args)

                if len(args) == 0 or cv == True:
                    file = f'./output/res_geo_{cv}_{"_".join(map(str, train_data.shape))}_{best_iteration}_{score:6.4f}_{"_".join(drop_list)}.csv'
                    res.to_csv(file)
                    gen_sub(file)
                else:
                    logger.debug('Search model, do not save file')

                feature_importance.to_hdf(f'./output/fi_{cv}_{best_iteration}_{"_".join(map(str, train_data.shape))}_{score:6.4f}_{"_".join(drop_list)}.h5',key='key')

    res = { 'loss': -score, 'status': STATUS_OK, 'attachments': {"message": f'{args} ', } }
    logger.info(res)
    return res

@timed()
def search():
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


if __name__ == '__main__':
    fire.Fire()
    # train_ex()
    # search()


""""
nohup python -u  core/train.py train_ex > train_profile_lda.log 2>&1 &

nohup python -u  core/train.py train_ex > train_geo_o.log 2>&1 &

nohup python -u  core/train.py train_ex > train_statistics_2.log 2>&1 &

nohup python -u  core/train.py search > search_logloss.log  2>&1 &

nohup python -u  core/train.py search > search_cv_sk.log  2>&1 &

nohup python -u  core/train.py train_ex > train_price_eta.log 2>&1 &

nohup python -u  core/train.py train_ex > train_price_eta_with_zero.log 2>&1 &

nohup python -u  core/train.py train_ex > remove_d_hash.log 2>&1 &

"""