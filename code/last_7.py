# -*- coding: utf-8 -*-
import numpy as np

import lightgbm as lgb
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from time import gmtime, strftime

def merge_raw_data():
    tr_queries = pd.read_csv('../data/train_queries.csv')
    te_queries = pd.read_csv('../data/test_queries.csv')
    tr_plans = pd.read_csv('../data/train_plans.csv')
    te_plans = pd.read_csv('../data/test_plans.csv')

    tr_click = pd.read_csv('../data/train_clicks.csv')

    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)

    te_data = te_queries.merge(te_plans, on='sid', how='left')
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def read_feature(data):
    for path in os.listdir('../pre_data'):
        temp_data = pd.read_csv('../pre_data/{}'.format(path))
        data = pd.concat([data,temp_data],axis=1)
    return data

def split_train_test(data):
    train_data = data[data['click_mode'] != -1]
    test_data = data[data['click_mode'] == -1]
    submit = test_data[['sid']].copy()
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, submit


def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def submit_result(submit, result, model_name):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(
        '../submit/{}_result_{}.csv'.format(model_name, now_time), index=False)

#最后七天验证集
def valid_lgb(train_x,train_y):
    data = pd.concat([train_x,pd.DataFrame(train_y,columns = ['click_mode'])],axis=1)
    train = data[data.req_time <='2018-11-23']
    valid = data[(data.req_time>'2018-11-23')&(data.req_time<'2018-12-01')]
    del data

    train_y = train[['click_mode']]
    print (train_y)
    train_x = train.drop(['click_mode', 'd', 'o', 'plan_time', 'plans', 'req_time'],axis=1)
    del train

    valid_y = valid[['click_mode']]
    valid_x = valid.drop(['click_mode', 'd', 'o', 'plan_time', 'plans', 'req_time'], axis=1)
    del valid

    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
    }

    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']

    train_set = lgb.Dataset(train_x, train_y, categorical_feature=cate_cols)
    val_set = lgb.Dataset(valid_x, valid_y, categorical_feature=cate_cols)

    lgb_model = lgb.train(lgb_paras, train_set,
                          valid_sets=[val_set], early_stopping_rounds=50, num_boost_round=40000, verbose_eval=50,
                          feval=eval_f)

    val_pred = np.argmax(lgb_model.predict(
        valid_x, num_iteration=lgb_model.best_iteration), axis=1)
    val_score = f1_score(valid_y, val_pred, average='weighted')
    print('cv f1-score: ', np.mean(val_score))


def train_lgb(train_x, train_y, test_x):
    train_x = train_x.drop(['req_time'], axis=1)
    test_x = test_x.drop(['req_time'], axis=1)


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
    }
    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']
    scores = []
    result_proba = []
    for tr_idx, val_idx in kfold.split(train_x, train_y):
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=cate_cols)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=cate_cols)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=50, num_boost_round=40000, verbose_eval=50, feval=eval_f)
        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        result_proba.append(lgb_model.predict(
            test_x, num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    return pred_test


if __name__ == '__main__':
    import feature
    train_x, train_y, test_x, submit = feature.get_train_test_feas_data()
    online = 0
    if online == 0:
        valid_lgb(train_x, train_y)
    else:
        result_lgb = train_lgb(train_x, train_y, test_x)
        submit_result(submit, result_lgb, 'lgb')