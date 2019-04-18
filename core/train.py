from core.feature import *

import lightgbm as lgb
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import f1_score


def gen_sub(file):
    #file = './output/res_False_0.6802.csv'
    res = pd.read_csv(file)
    res.sid = res.sid.astype(object)
    res = res.set_index('sid')
    res['recommend_mode'] = res.iloc[:, 1:].idxmax(axis=1)
    import csv
    print(csv.QUOTE_NONE)
    sub_file  = file.replace('res', 'sub')
    res[['recommend_mode']].to_csv(sub_file, quoting=csv.QUOTE_ALL, header=None)
    logger.info(f'Sub file save to {sub_file}')


@timed()
def train(X_data, y_data, X_test, cv=False):
    num_fold = 5
    num_class = 12
    folds = KFold(n_splits=num_fold, shuffle=True, random_state=15)
    oof = np.zeros((len(y_data), num_class))
    predictions = np.zeros((len(X_test), num_class))
    # start = time.time()
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data.values, y_data.values)):
        logger.info("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X_data.iloc[trn_idx], y_data.iloc[trn_idx])
        val_data = lgb.Dataset(X_data.iloc[val_idx], y_data.iloc[val_idx], reference=trn_data)

        # np.random.seed(666)
        params = {
            'verbose':-1,
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 4,
            'objective': 'multiclass',
            'num_class': num_class,
        # lightgbm.basic.LightGBMError: b'Number of classes should be specified and greater than 1 for multiclass training'
            # 'device':'gpu',
            'gpu_platform_id': 1, 'gpu_device_id': 0
        }
        num_round = 30000
        clf = lgb.train(params,
                        trn_data,
                        num_round,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=200,
                        early_stopping_rounds=200)

        oof[val_idx] = clf.predict(X_data.iloc[val_idx], num_iteration=clf.best_iteration)

        score = f1_score(y_data.iloc[val_idx].values, oof[val_idx].argmax(axis=1), average='weighted')

        logger.info(f'fold n{fold_}, local score:{score:6.4f},best_iter:{clf.best_iteration}, val shape:{X_data.iloc[val_idx].shape}')

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
                            verbose_eval=2000,
                            )
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
            logger.info('CV is disable, will train with full train data with iter:{clf.best_iteration}')
            break
    predictions = predictions / (fold_ + 1)
    if cv:
        score = f1_score(y_data.values, oof.argmax(axis=1), average='weighted')

    logger.info(f'cv:{cv}, the final local score:{score:6.4f}, predictions:{predictions.shape}')
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=range(12))
    predictions.index.name = 'sid'
    return predictions, score



if __name__ == '__main__':
    feature = get_feature(('profile'))  # .fillna(0)
    train_data = feature.loc[feature.click_mode >= 0]
    X_data, y_data = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test = feature.loc[feature.click_mode == -1].iloc[:, :-1]

    res, score = train(X_data, y_data, X_test, cv=False)
    res.to_csv(f'./output/res_{feature.shape[1]}_False_{score:6.4f}.csv')

    res, score = train(X_data, y_data, X_test, cv=True)
    res.to_csv(f'./output/res_{feature.shape[1]}_True_{score:6.4f}.csv')


    feature = get_feature(None)#.fillna(0)
    train_data = feature.loc[feature.click_mode>=0]
    X_data,  y_data = train_data.iloc[:,:-1], train_data.iloc[:,-1]
    X_test = feature.loc[feature.click_mode==-1].iloc[:,:-1]

    res, score = train(X_data, y_data, X_test, cv=False )
    res.to_csv(f'./output/res_{feature.shape[1]}_False_{score:6.4f}.csv')

    res, score = train(X_data, y_data, X_test, cv=True)
    res.to_csv(f'./output/res_{feature.shape[1]}_True_{score:6.4f}.csv')