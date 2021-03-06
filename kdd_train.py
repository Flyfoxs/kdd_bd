import sys
sys.path.append('./')
from ph3.kdd_phase3_refactor import *


@timed()
def train():
    #Gen feature in a separator process to recycle memory before training
    from multiprocessing import Process
    p = Process(target=gen_feature, name='get_feature')
    p.start()
    p.join()
    gc.collect()
    oof_file=train_base()
    gc.collect()
    #gen_sub(oof_file)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
        self.coef_arr = []
        self.best_score = 0
        self.initial_score = 0
        self.val_score = []
        self.initial_coef = [1.17117383, 0.94684846, 0.69123502, 2.46440181, 3.24613048,
       0.81681875, 1.8995724 , 0.82457647, 1.47187443, 0.84245455,
       1.24738773, 0.98517357]

    def _kappa_loss(self, coef, X, y):
        X_p = DF(np.copy(X))
        for i in range(len(coef)):
            X_p[i] *= coef[i]

        l1 = f1_score(y, np.argmax(X_p.values, axis=1), average="weighted")
        self.coef_arr.append(coef)
        self.best_score = max(l1, self.best_score)
        print(list(coef.astype(np.float16)), ' Train score = ', l1.astype(np.float32), 'Best Score',
              self.best_score)  # ,' Valid score =',l2.astype(np.float16))
        return -l1

    def fit(self, X, y):
        self.initial_score = f1_score(y, np.argmax(X.values, axis=1), average="weighted")

        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(loss_partial, self.initial_coef, method='Powell')

    def predict(self, X, coef):

        logger.info(f'Predict with:{coef}')
        #print(self.coef_)
        X_p = DF(np.copy(X))
        for i in range(len(coef)):
            X_p[i] *= coef[i]
        return X_p

    def coefficients(self):
        return self.coef_['x']


@timed()
def gen_sub(oof_file):
    opt = OptimizedRounder()
    train = pd.read_hdf(oof_file, 'train')
    test = pd.read_hdf(oof_file, 'test')
    opt.fit(train.iloc[:, :12], train.iloc[:, 12].astype(int))

    for coef, name in [(opt.coef_['x'], 'best'),  (opt.initial_coef, 'default')]:
        logger.info(f'Adjust with coef:{coef}')
        test_pred = opt.predict(test.iloc[:, :12], coef)
        test_pred = np.argmax(test_pred.values, axis=1)
        test_pred = pd.DataFrame(test_pred, columns=['recommend_mode'], index=test.index)
        test_pred.index.name = 'sid'
        sub_file = f'./result/{name}_{opt.initial_score:6.5f}_{opt.best_score:6.5f}.csv'
        test_pred.to_csv(sub_file)
        logger.info(f'Sub file save to:{sub_file}')
    logger.info(f'Best coef is: {opt.coefficients()}')
    return opt.coefficients()

@timed()
def reduce_by_svd(df, n_component):
    svd = TruncatedSVD(n_components=n_component, random_state=2019)
    df = df.fillna(0)
    return svd.fit_transform(df.values)



@timed()
def train_base():

    all_data = get_feature_all()
    all_data = convert_int32_float32(all_data)

    logger.info(f'Current all_data dtypes:\n {all_data.dtypes.value_counts()}')
    #all_data = all_data.sample(frac=0.1)

    #all_data = reduce_dim(all_data)

    try:
        logger.info(f'cache_clear:Cache info for get_plans:{get_plans.cache_info()}')
        get_plans.cache_clear()
    except AttributeError as e:
        logger.info(f'cache_clear:No Cache for fun#get_plans')


    # Define F1 Train
    feature_name = get_feature_name(all_data)
    logger.info(f'Final Train feature#{len(feature_name)}: {sorted(feature_name)}')
    # CV TRAIN

    tr_index = ~all_data['click_mode'].isnull()
    y = all_data[tr_index]['click_mode'].reset_index(drop=True)

    # # all_data = reduce_by_svd(all_data[list(set(feature_name))], 500)
    #
    # all_data = pd.DataFrame(all_data).add_prefix('svd_')
    # feature_name = all_data.columns

    X_train = all_data[tr_index]#[list(set(feature_name))].reset_index(drop=True)
    X_test = all_data[~tr_index]#[list(set(feature_name))].reset_index(drop=True)
    del all_data
    print(X_train.shape, X_test.shape)
    final_pred = []
    cv_score = []
    cv_model = []
    skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):

        with timed_bolck(f'CV Train#{index},feature:{len(feature_name)}'):
            gc.collect()
            # with timed_bolck(f'Folder#{index}, feature:{len(feature_name)}'):
            lgb_model = lgb.LGBMClassifier(
                boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
                max_depth=-1, n_estimators=3000, objective='multiclass', num_classes=12,
                subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
                learning_rate=0.1, random_state=2019 + index, n_jobs=6, metric="None", importance_type='gain'
            )


            logger.info(f'Begin to split X_Train:{X_train[feature_name].shape}')
            train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[
                test_index], y.iloc[train_index], y.iloc[test_index]
            eval_set = [(test_x[feature_name].values, test_y.values)]


            logger.info(f'Begin Train#{index}, feature:{len(feature_name)}, Size:{train_x[feature_name].shape}')
            train_x = train_x[feature_name].values
            train_y = train_y.values
            logger.info(f'train_x:{train_x.dtype}')

            gc.collect()
            lgb_model.fit(train_x, train_y, eval_set=eval_set, verbose=10, early_stopping_rounds=30,
                          eval_metric=f1_macro)

            cv_model.append(lgb_model)

            from sklearn.externals import joblib
            # save model
            score = lgb_model.best_score_["valid_0"]["f1_score"]
            mode_file = f'./model/model_normal_{len(feature_name)}_{index}_{lgb_model.best_iteration_}_{score:6.5f}.pkl'
            joblib.dump(lgb_model, mode_file )

            lgb_model.booster_.save_model(f'./model/model_normal_{len(feature_name)}_{index}.txt')


            y_test = lgb_model.predict(X_test[feature_name])
            y_val = lgb_model.predict_proba(test_x[feature_name])
            cur_score = get_f1_score(test_y, y_val)
            logger.info(f'End Train#{index}, cur_score:<<{cur_score:6.5f}>>, best_iter:{lgb_model.best_iteration_}')
            cv_score.append(cur_score)
            print(Counter(np.argmax(y_val, axis=1)))

            if index == 0:
                final_pred = np.array(y_test).reshape(-1, 1)
            else:
                final_pred = np.hstack((final_pred, np.array(y_test).reshape(-1, 1)))

            del train_x, test_x, train_y, test_y, eval_set
            gc.collect()

    cv_pred = np.zeros((X_train.shape[0], 12))
    test_pred = np.zeros((X_test.shape[0], 12))
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], \
                                           y.iloc[test_index]
        y_val = cv_model[index].predict_proba(test_x[feature_name])
        print(y_val.shape)
        cv_pred[test_index] = y_val
        test_pred += cv_model[index].predict_proba(X_test[feature_name]) / 5

    #print(np.mean(cv_score))

    oof_train = DF(cv_pred)
    all_data = get_feature_stable()[['sid', 'click_mode']]
    # oof_train.columns = ['label_'+str(i) for i in range(0,12)]
    oof_train['sid'] = all_data[all_data['click_mode'].notnull()]['sid'].values
    oof_train[12] = y
    # oof_train['click_mode'] = all_data[tr_index]['click_mode'].reset_index(drop=True)
    oof_train.set_index('sid', inplace=True)

    oof_test = DF(test_pred)
    # oof_test.columns = ['label_'+str(i) for i in range(0,12)]
    oof_test['sid'] = all_data[~tr_index]['sid'].values
    oof_test[12] = np.nan
    oof_test.set_index('sid', inplace=True)

    avg_score = np.mean(cv_score)
    oof_file = f"./output/stacking/oof_{cv}_fold_{test_x.shape[1]}_{avg_score}_feature_phase2.hdf"
    oof_train.to_hdf(oof_file, 'train')
    oof_test.to_hdf(oof_file, 'test')

    logger.info(f'Avg score:{avg_score}, OOF save to :{oof_file}')

    return oof_file





#
# def lgb_f1_score_avg(y_hat, data, average):
#     y_true = data.get_label()
#     num_sample = len(y_true)
#     # print(y_hat.shape, y_hat[:10])
#     if average == 'binary':
#         y_hat = [0 if item < 0.3 else 1 for item in y_hat]
#         # y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
#         # print(y_true.shape, y_hat.shape, y_true[:10], y_hat[:10])
#         score = f1_score(y_true, y_hat, average=average)
#         return 'f1', round(score, 4), True
#     else:
#         y_hat = y_hat.reshape(-1, num_sample).T.argmax(axis=1)
#         # y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
#         # print(y_true.shape, y_hat.shape, y_true[:10], y_hat[:10])
#         score = f1_score(y_true, y_hat, average=average)
#         return 'f1', round(score, 4), True
#
#
# lgb_f1_score = partial(lgb_f1_score_avg, average='weighted')
#
#
# @timed()
# def train_lgb():
#     all_data = get_feature_all()
#     feature_name = get_feature_name(all_data)
#
#     all_data = all_data.loc[pd.notna(all_data.click_mode)]
#
#     skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
#     for fold_, (train_index, test_index) in enumerate(skf.split(all_data, all_data.click_mode)):
#         gc.collect()
#         with timed_bolck(f'Fold#{fold_}'):
#             trn_data = lgb.Dataset(all_data[feature_name].iloc[train_index], all_data.click_mode.iloc[train_index])
#             val_data = lgb.Dataset(all_data[feature_name].iloc[test_index], all_data.click_mode.iloc[test_index],
#                                    reference=trn_data)
#
#             params = {
#                 'nthread': -1,
#                 'verbose': -1,
#                 'num_leaves': 128,
#                 ### 'min_data_in_leaf': 90,
#                 'feature_fraction': 0.5,
#                 'lambda_l1': 0.1,
#                 'lambda_l2': 10,
#                 'max_depth': 6,
#
#                 'learning_rate': 0.1,
#                 'bagging_fraction': 0.7,
#
#                 'objective': 'multiclass',
#                 'metric': 'None',
#                 'num_class': 12,
#                 # 'random_state': 2019,
#                 # 'device':'gpu',
#                 # 'gpu_platform_id': 1, 'gpu_device_id': 0
#             }
#             # params = dict(params, **args)
#
#             logger.info(params)
#
#             num_round = 30000
#             # num_round = 10
#             verbose_eval = 50
#             clf = None
#             with timed_bolck(f'Train#{fold_}'):
#                 clf = lgb.train(params,
#                                 trn_data,
#                                 num_round,
#                                 valid_sets=[trn_data, val_data],
#                                 feval=lgb_f1_score,  # lgb_f1_score,
#                                 verbose_eval=verbose_eval,
#                                 early_stopping_rounds=400)
#



if __name__ == '__main__':
    """
    运行方式:
    nohup python -u kdd_train.py train_base 50 &
    nohup python -u kdd_train.py train > train_28.log 2>&1  &
    
    python -u kdd_train.py train_base

    快速测试代码逻辑错: 
    get_queries,里面的采样比例即可

    """
    fire.Fire()


