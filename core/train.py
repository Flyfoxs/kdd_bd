from core.feature import *

import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    num_sample = len(y_true)
    #print(y_hat.shape, y_hat[:10])
    y_hat =  y_hat.reshape(-1, num_sample).T.argmax(axis=1)
    #y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    #print(y_true.shape, y_hat.shape, y_true[:10], y_hat[:10])
    score = f1_score(y_true, y_hat, average='weighted')
    return 'f1', round(score, 4) , True


def gen_sub(file):
    # file = './output/res_False_0.6802.csv'
    res = pd.read_csv(file)
    res.sid = res.sid.astype(object)
    res = res.set_index('sid')
    res['recommend_mode'] = res.idxmax(axis=1)

    # Fix the data if no plan at all
    feature = get_feature()  # .fillna(0)
    #feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]
    adjust_sid = feature.loc[(feature.click_mode == -1) & (feature.o_seq_0_ == 0)]
    len_prepare_adjust = len(adjust_sid)
    real_need_to_adjust = len(res.loc[adjust_sid.index])
    if  len_prepare_adjust != real_need_to_adjust:
        raise Exception(f'Error when adjust 0 model, prepare:{len_prepare_adjust}, real:{real_need_to_adjust}')
    logger.info(f'Manually adjust these records:{len_prepare_adjust}, the original value is:')
    logger.info(res.loc[adjust_sid.index, 'recommend_mode'].value_counts())

    res.loc[adjust_sid.index, 'recommend_mode'] = 0

    import csv
    sub_file = file.replace('res', 'sub/sub')
    res[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL, header=None)
    logger.info(f'Sub file save to {sub_file}')


def get_groups(cut_point = val_cut_point):
    feature = get_feature()
    #feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]
    feature = feature.loc[(feature.click_mode >= 0) & (feature.o_seq_0_ > 0)].reset_index(drop=True)
    day_ = feature.date
    day_ = day_ - min(day_)
    day_ = day_.dt.days
    #end = max(day_)
    return [(day_.loc[day_<=cut_point].index.values ,day_.loc[day_>cut_point].index.values) ]


@timed()
def train_lgb(X_data, y_data, X_test, cv=False, args={}):

    num_class = 12

    oof = np.zeros((len(y_data), num_class))
    predictions = np.zeros((len(X_test), num_class))
    # start = time.time()
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(get_groups()):
        logger.info(f"fold n°{fold_}, cv:{cv},train:{trn_idx.shape}, val:{val_idx.shape} " )
        trn_data = lgb.Dataset(X_data.iloc[trn_idx], y_data.iloc[trn_idx])
        val_data = lgb.Dataset(X_data.iloc[val_idx], y_data.iloc[val_idx], reference=trn_data)

        # np.random.seed(666)
        params = {
            'verbose':-1,
            'num_leaves': 70,
            'min_data_in_leaf': 149,
            'feature_fraction':0.65,
            'lambda_l1': 0.9,
            'lambda_l2': 1,
            'max_depth': 6,

            'learning_rate': 0.1,
            'bagging_fraction': 0.7,
            # 'reg_alpha': 0.7,
            # 'reg_lambda': 1,

            'objective': 'multiclass',
            'metric': 'None',
            'num_class': num_class,
        # lightgbm.basic.LightGBMError: b'Number of classes should be specified and greater than 1 for multiclass training'
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

        oof[val_idx] = clf.predict(X_data.iloc[val_idx], num_iteration=clf.best_iteration)

        score = f1_score(y_data.iloc[val_idx].values, oof[val_idx].argmax(axis=1), average='weighted')

        logger.info(f'fold n{fold_}, cv:{cv}, local score:{score:6.4f},best_iter:{clf.best_iteration}, val shape:{X_data.iloc[val_idx].shape}')

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_data.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if cv:
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
        elif len(args)==0: #not Search model
            all_train = lgb.Dataset(X_data, y_data)
            clf = lgb.train(params,
                            all_train,
                            # num_round,
                            num_boost_round=clf.best_iteration,
                            valid_sets=[all_train],
                            feval=lgb_f1_score,
                            verbose_eval=verbose_eval,
                            )
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
            logger.info(f'CV is disable, will train with full train data with iter:{clf.best_iteration}')
            break
    predictions = predictions / (fold_ + 1)
    if cv:
        score = f1_score(y_data.values, oof.argmax(axis=1), average='weighted')

    logger.info(f'cv:{cv}, the final local score:{score:6.4f}, predictions:{predictions.shape}, params:{params}')
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=[str(i) for i in range(12)])
    predictions.index.name = 'sid'
    return predictions, score, feature_importance_df


def plot_import(feature_importance):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(16, 12));
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
    plt.title('LGB Features (avg over folds)')

from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
def get_search_space():
    space = {"num_leaves":hp.choice("num_leaves", range(70, 100, 10)),
             #"max_depth": hp.choice("max_depth", [7]),
             'lambda_l1': hp.choice("lambda_l1", [0.9, 1, 10, 20]),
             'lambda_l2': hp.choice("lambda_l2", [0.9, 1, 2, 5]),
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
        ['date'],
    ]):

        for ratio in range(1):
            feature = get_feature()  # .fillna(0)
            feature = feature.drop(drop_list,axis=1,errors='ignore')

            logger.info((feature.shape, list(feature.columns)))

            for col, type_ in feature.dtypes.sort_values().iteritems():
                if type_ not in ['int64', 'int16', 'int32', 'float64']:
                    logger.error(col, type_)

            feature.columns = ['_'.join(item) if isinstance(item, tuple) else item for item in feature.columns]

            train_data = feature.loc[(feature.click_mode >= 0) & (feature.o_seq_0_ >0)]
            X_data, y_data = train_data.iloc[:, :-1], train_data.iloc[:, -1]

            X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

            for cv in [False]:
                res, score, feature_importance = train_lgb(X_data, y_data, X_test, cv=cv, args=args)

                file = f'./output/res_ratio_{ratio:3.1f}_{sn}_{feature.shape[1]}_{cv}_{score:6.4f}_{"_".join(drop_list)}.csv'
                res.to_csv(file)
                gen_sub(file)

                feature_importance.to_hdf(f'./output/fi_drop_sn_{sn},_{feature.shape[1]}_{cv}_{score:6.4f}_{"_".join(drop_list)}.h5',key='key')

    res = { 'loss': -score, 'status': STATUS_OK, 'attachments': {"message": f'{args} ', } }
    logger.info(res)
    return res

@timed()
def search():
    trials = Trials()
    space = get_search_space()
    best = fmin(train_ex, space, algo=tpe.suggest, max_evals=30, trials=trials)

    logger.debug(f"Best: {best}")
    att_message = [trials.trial_attachments(trial)['message'] for trial in trials.trials]
    for score, para, misc in zip(trials.losses(),
                                 att_message,
                                 [item.get('misc').get('vals') for item in trials.trials]
                                 ):
        logger.debug(f'score:{"%9.6f"%score}, para:{para}, misc:{misc}')


if __name__ == '__main__':
    #train_ex()
    search()


""""
nohup python -u  core/train.py > search_5.log 2>&1 &

nohup python -u  core/train.py > sub.log 2>&1 &
"""