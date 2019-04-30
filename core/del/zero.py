from core.feature import *
import lightgbm as lgb
from core.split import *
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
@timed()
def train_lgb(X_data, y_data, X_test, cv=False, args={}):

    num_class = 0

    oof = np.zeros((len(y_data)))
    predictions = np.zeros((len(X_test)))
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
        logger.info(f"fold n°{fold_}, cv:{cv},train:{trn_idx.shape}, val:{val_idx.shape}, test:{X_test.shape}, cat:{cate_cols} " )
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

            'objective': 'binary',
            'metric': 'binary_logloss',
            #'num_class': num_class,
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
                        feval=lgb_f1_score_bin,
                        verbose_eval=verbose_eval,
                        early_stopping_rounds=400)

        max_iteration = max(max_iteration, clf.best_iteration)
        min_iteration = min(min_iteration, clf.best_iteration)

        oof[val_idx] = clf.predict_proba(X_data.iloc[val_idx], num_iteration=clf.best_iteration)

        score = f1_score(y_data.iloc[val_idx].values, oof[val_idx].argmax(axis=1), average='weighted')

        logger.info(f'fold n{fold_}, cv:{cv}, local score:{score:6.4f},best_iter:{clf.best_iteration}, val shape:{X_data.iloc[val_idx].shape}')

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_data.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if cv:
            predictions += clf.predict_proba(X_test, num_iteration=clf.best_iteration)
        elif len(args)==0: #not Search model
            all_train = lgb.Dataset(X_data, y_data, categorical_feature=cate_cols)
            clf = lgb.train(params,
                            all_train,
                            # num_round,
                            num_boost_round=clf.best_iteration,
                            valid_sets=[all_train],
                            feval=lgb_f1_score_bin,
                            verbose_eval=verbose_eval * 2,
                            )
            predictions += clf.predict_proba(X_test, num_iteration=clf.best_iteration)
            logger.info(f'CV is disable, will train with full train data with iter:{clf.best_iteration}')
            break
    predictions = predictions / (fold_ + 1)
    if cv:
        score = f1_score(y_data.values, oof.argmax(axis=1), average='weighted')

    logger.info(f'cv:{cv}, the final local score:{score:6.4f}, predictions:{predictions.shape}, params:{params}')
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=[str(i) for i in range(12)])
    predictions.index.name = 'sid'
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return predictions, score, feature_importance_df, f'{min_iteration}_{max_iteration}'

def train_ex(args={}):
    for ratio in range(1):
        train_data, X_test = get_train_test_zero()

        X_data, y_data = train_data.iloc[:, :-1], train_data.iloc[:, -1]

        for cv in [True,]:
            res, score, feature_importance, best_iteration = train_lgb(X_data, y_data, X_test, cv=cv, args=args)

            if len(args) == 0 or cv == True:
                file = f'./output/zero_{cv}_{train_data.shape[1]}_{best_iteration}_{score:6.4f}_{"_".join(drop_list)}.csv'
                res.to_csv(file)
                #gen_sub(file)
            else:
                logger.debug('Search model, do not save file')

            feature_importance.to_hdf(f'./output/fi_zero_{cv}_{best_iteration}_{train_data.shape[1]}_{score:6.4f}_{"_".join(drop_list)}.h5',key='key')

    res = { 'loss': -score, 'status': STATUS_OK, 'attachments': {"message": f'{args} ', } }
    logger.info(res)
    return res


if __name__ == '__main__':
    import fire
    fire.Fire()

    """
 
    nohup python -u  core/zero.py train_ex > fm.log  2>&1 &
    """