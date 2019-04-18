from core.feature import *

import lightgbm as lgb
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    num_sample = len(y_true)
    y_hat =  y_hat.reshape(-1, num_sample).T.argmax(axis=1)
    #y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    score = f1_score(y_true, y_hat, average='weighted')
    return 'f1', round(score, 4) , True


def gen_sub(file):
    #file = './output/res_False_0.6802.csv'
    res = pd.read_csv(file)
    res.sid = res.sid.astype(object)
    res = res.set_index('sid')
    res['recommend_mode'] = res.iloc[:, 1:].idxmax(axis=1)
    import csv
    print(csv.QUOTE_NONE)
    sub_file  = file.replace('res', '/sub/sub')
    res[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL, header=None)
    logger.info(f'Sub file save to {sub_file}')


def get_groups():
    train = get_original('train_queries.csv')
    day_ = pd.to_datetime(train.req_time).dt.date
    day_ = day_ - min(day_)
    day_ = day_.dt.days
    end = max(day_)
    cut_point = [end-7, end-14]
    return [(day_.loc[day_<=point].index.values ,day_.loc[day_>point].index.values) for point in cut_point]


@timed()
def train(X_data, y_data, X_test, cv=False):

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
            'num_leaves': 111,
            'min_data_in_leaf': 149,
            'feature_fraction':0.8,
            'bagging_fraction':0.7,
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 4,
            'objective': 'multiclass',
            'metric': 'None',
            'num_class': num_class,
        # lightgbm.basic.LightGBMError: b'Number of classes should be specified and greater than 1 for multiclass training'
            # 'device':'gpu',
            #'gpu_platform_id': 1, 'gpu_device_id': 0
        }
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
        else:

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

    logger.info(f'cv:{cv}, the final local score:{score:6.4f}, predictions:{predictions.shape}')
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=range(12))
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


if __name__ == '__main__':

    if __name__ == '__main__':
        for group in [('profile'), None, ]:
            feature = get_feature(group)  # .fillna(0)
            train_data = feature.loc[feature.click_mode >= 0]
            X_data, y_data = train_data.iloc[:, :-1], train_data.iloc[:, -1]
            X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

            for cv in [False]:
                res, score, feature_importance = train(X_data, y_data, X_test, cv=cv)
                file = f'./output/res_7days_{feature.shape[1]}_{cv}_{score:6.4f}.csv'
                res.to_csv(file)
                gen_sub(file)

                feature_importance.to_hdf('./output/fi_{feature.shape[1]}_{cv}_{score:6.4f}.h5',key='key')



""""
nohup python -u  core/train.py > 7days2.log 2>&1 &
"""