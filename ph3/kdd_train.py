import sys
sys.path.append('./')
from ph3.kdd_phase3_refactor import *


def train():
    pass


@timed()
def train_base(feature_cnt=9999):
    import sys
    print(sys.path)
    all_data = get_feature_all()#.sample(frac=0.2)
    # Define F1 Train
    feature_name = get_feature_name(all_data)[:feature_cnt]
    logger.debug(f'Final Train feature#{len(feature_name)}: {sorted(feature_name)}')
    # CV TRAIN

    tr_index = ~all_data['click_mode'].isnull()
    X_train = all_data[tr_index][list(set(feature_name))].reset_index(drop=True)
    y = all_data[tr_index]['click_mode'].reset_index(drop=True)
    X_test = all_data[~tr_index][list(set(feature_name))].reset_index(drop=True)
    print(X_train.shape, X_test.shape)
    final_pred = []
    cv_score = []
    cv_model = []
    skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        with timed_bolck(f'Begin Train#{index},feature:{len(feature_name)}'):
            gc.collect()
            # with timed_bolck(f'Folder#{index}, feature:{len(feature_name)}'):
            lgb_model = lgb.LGBMClassifier(
                boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
                max_depth=-1, n_estimators=3000, objective='multiclass', num_classes=12,
                subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
                learning_rate=0.1, random_state=2019 + index, n_jobs=6, metric="None", importance_type='gain'
            )
            train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[
                test_index], y.iloc[train_index], y.iloc[test_index]
            eval_set = [(test_x[feature_name], test_y)]
            logger.info(f'Begin Train#{index}, feature:{len(feature_name)}, Size:{train_x[feature_name].shape}')
            lgb_model.fit(train_x[feature_name], train_y, eval_set=eval_set, verbose=10, early_stopping_rounds=30,
                          eval_metric=f1_macro)

            cv_model.append(lgb_model)
            y_test = lgb_model.predict(X_test[feature_name])
            y_val = lgb_model.predict_proba(test_x[feature_name])
            cur_score = get_f1_score(test_y, y_val)
            logger.info(f'End Train#{index}, cur_score:<<{cur_score:6.5}>>, best_iter:{lgb_model.best_iteration_}')
            cv_score.append(cur_score)
            print(Counter(np.argmax(y_val, axis=1)))

            if index == 0:
                final_pred = np.array(y_test).reshape(-1, 1)
            else:
                final_pred = np.hstack((final_pred, np.array(y_test).reshape(-1, 1)))
    # fi = []
    # for i in cv_model:
    #     tmp = {
    #         'name': feature_name,
    #         'score': i.feature_importances_
    #     }
    #     fi.append(pd.DataFrame(tmp))
    #
    # fi = pd.concat(fi)
    #
    # fi.groupby(['name'])['score'].agg('mean').sort_values(ascending=False).head(30).plot.barh()

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
    oof_file = f"./output/stacking/oof_{cv}_fold_{len(feature_name)}_{avg_score}_feature_phase2.hdf"
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

def adjust_after(oof_file):
    pass

if __name__ == '__main__':
    """
    运行方式:
    nohup python ph3/kdd_train.py train_base 50 &
    nohup python ph3/kdd_train.py train_base > train.log 2>&1  &

    快速测试代码逻辑错: 
    get_queries,里面的采样比例即可

    """
    fire.Fire()




    #
    # # Offline 后处理前
    #
    # if version == 2:
    #     train_clicks_2 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version),parse_dates=['click_time'],nrows=nrows)
    #     train_clicks_1 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version-1),parse_dates=['click_time'],nrows=nrows)
    #     answer = train_clicks_2.append(train_clicks_1).reset_index(drop=True).fillna(0)
    # else:
    #     answer = pd.read_csv(input_dir+'train_clicks.csv',parse_dates=['click_time'],nrows=nrows)
    #
    # if offline:
    #     answer = all_data[~tr_index][['sid','city']].merge(answer,how='left',on='sid').fillna(0)
    #     answer['pred'] = np.argmax(test_pred,axis=1)
    #     print(f1_score(answer['click_mode'],answer['pred'],average='weighted'))
    #
    # for i in range(0,4):
    #     tmp = answer[answer['city']==i]
    #     print(i,f1_score(tmp['click_mode'],tmp['pred'],average='weighted'))
    #
    # num_classes = 12
    # label_name = 'click_mode'
    # oof_train.rename(columns={num_classes:label_name,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11},inplace=True)
    # oof_test.rename(columns={num_classes:label_name,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11},inplace=True)
    #
    # tmp_train = oof_train[oof_train.index.isin(space_time[space_time['req_time']>='2018-11-15']['sid'].unique())]
    # sz_train = tmp_train[tmp_train.index.isin(space_time[space_time['city']==2]['sid'].unique())]
    # other_train = tmp_train.copy()#[tmp_train.index.isin(space_time[space_time['city']!=2]['sid'].unique())]
    # raw_train_score = f1_score(tmp_train[label_name],np.argmax(tmp_train[range(num_classes)].values,axis=1),average='weighted')
    # #raw_valid_score = f1_score(valid[label_name],np.argmax(valid[range(num_classes)].values,axis=1),average='weighted')
    #
    # print("RAW SCORE: ",raw_train_score)#raw_valid_score
    #
    # class OptimizedRounder(object):
    #     def __init__(self):
    #         self.coef_ = 0
    #         self.coef_arr = []
    #         self.val_score = []
    #
    #     def _kappa_loss(self, coef, X, y):
    #         X_p = DF(np.copy(X))
    #         for i in range(len(coef)):
    #             X_p[i] *= coef[i]
    #
    #         l1 = f1_score(y, np.argmax(X_p.values,axis=1), average="weighted")
    #         self.coef_arr.append(coef)
    #
    #         print(list(coef.astype(np.float16)),' Train score = ',l1.astype(np.float32))#,' Valid score =',l2.astype(np.float16))
    #         return -l1
    #
    #     def fit(self, X, y):
    #         loss_partial = partial(self._kappa_loss, X=X, y=y)
    #         self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Powell')
    #
    #     def predict(self, X, coef):
    #         X_p = DF(np.copy(X))
    #         for i in range(len(coef)):
    #             X_p[i] *= coef[i]
    #         return X_p
    #
    #     def coefficients(self):
    #         return self.coef_['x']
    #
    # # 下面是一起处理的
    # # cv_pred = tmp_train[range(num_classes)].values
    # # y = tmp_train[label_name].values
    # # initial_coef = [1.0000] * num_classes
    #
    # # optR = OptimizedRounder()
    # # optR.fit(cv_pred, y)
    # # best_score = optR.coefficients()
    #
    # # best_coef = optR.coefficients()
    # # print(best_coef)#,best_score
    #
    # # SZ
    #
    # cv_pred = sz_train[range(num_classes)].values
    # y = sz_train[label_name].values
    # initial_coef = [1.1] * num_classes
    #
    # optR = OptimizedRounder()
    # optR.fit(cv_pred, y)
    # sz_score = optR.coefficients()
    #
    # # Other
    #
    # cv_pred = other_train[range(num_classes)].values
    # y = other_train[label_name].values
    # initial_coef = [1.1] * num_classes
    #
    # optR = OptimizedRounder()
    # optR.fit(cv_pred, y)
    # other_score = optR.coefficients()
    #
    # sz_test = oof_test[oof_test.index.isin(space_time[space_time['city']==2]['sid'].unique())]
    # other_test = oof_test[oof_test.index.isin(space_time[space_time['city']!=2]['sid'].unique())]
    # print(sz_test.shape,other_test.shape)
    #
    # sz_y = list(sz_train[label_name].values)
    # other_y = list(other_train[label_name].values)
    # y = sz_y + other_y
    #
    # sz_train = optR.predict(sz_train[range(num_classes)].values,sz_score)
    # other_train = optR.predict(other_train[range(num_classes)].values,other_score)
    # cv_pred = sz_train.append(other_train)
    #
    # print("Global Best")
    # print(best_coef)
    # print("\nValid Counts = ", Counter(y))
    # print("Predicted Counts = ", Counter(np.argmax(cv_pred.values,axis=1)))
    # acc1 = raw_train_score
    # acc2 = f1_score(y,np.argmax(cv_pred.values,axis=1),average="weighted")
    # print("Train Before = ",acc1)
    # print("Train After = ",acc2)
    # print("Train GAP = ",acc2-acc1)
    #
    # test_pred = optR.predict(oof_test[range(num_classes)], best_coef)
    # test_pred = np.argmax(test_pred.values,axis=1)
    #
    # # 后处理后
    #
    # if version == 2:
    #     train_clicks_2 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version),parse_dates=['click_time'],nrows=nrows)
    #     train_clicks_1 = pd.read_csv(input_dir+'train_clicks_phase{}.csv'.format(version-1),parse_dates=['click_time'],nrows=nrows)
    #     answer = train_clicks_2.append(train_clicks_1).reset_index(drop=True).fillna(0)
    # else:
    #     answer = pd.read_csv(input_dir+'train_clicks.csv',parse_dates=['click_time'],nrows=nrows)
    #
    # if offline:
    #     answer = all_data[~tr_index][['sid','city']].merge(answer,how='left',on='sid').fillna(0)
    #     answer['pred'] = test_pred
    #     print(f1_score(answer['click_mode'],answer['pred'],average='weighted'))
    #
    # for i in range(0,4):
    #     tmp = answer[answer['city']==i]
    #     print(i,f1_score(tmp['click_mode'],tmp['pred'],average='weighted'))
    #
    # ALL