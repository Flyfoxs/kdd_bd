from core.feature import *

import lightgbm as lgb
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import f1_score



@timed()
def train(X_data, y_data, X_test):
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
            'verbose':0,
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

        val_score = f1_score(y_data[val_idx], oof[val_idx].argmax(axis=1), average='weighted')

        logger.info(f'fold n{fold_}, score:{val_score:6.4f},best_iter:{clf.best_iteration}, val shape:{X_data.iloc[val_idx].shape}')

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_data.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(X_test, num_iteration=clf.best_iteration)
    predictions = predictions / folds.n_splits
    oof = oof.argmax(axis=1)
    score = f1_score(y_data.values, oof, average='weighted')
    logger.info(f'The final local score:{score:6.4f}')
    return predictions, score

if __name__ == '__main__':
    feature = get_feature()
    train_data = feature.loc[feature.click_mode>=0]
    X_data,  y_data = train_data.iloc[:,:-1], train_data.iloc[:,-1]
    X_test = feature.loc[feature.click_mode==-1].iloc[:,:-1]
    res, score = train(X_data,  y_data,  X_test )