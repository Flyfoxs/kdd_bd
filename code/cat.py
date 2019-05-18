# -*- coding: utf-8 -*-
import numpy as np
import feature
import catboost as cat
import lightgbm as lgb
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from time import gmtime, strftime
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder



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
    data = data.drop(['plan_time', 'd', 'o', 'plans', 'req_time'], axis=1)
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

def merge_result(submit, result, model_name):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    result = pd.DataFrame(result)
    result.columns = [index for index in range(12)]
    submit = pd.concat([submit,result],axis=1)
    submit.to_csv(
        '../merge/merge_{}_result_{}.csv'.format(model_name, now_time), index=False)


def valid_cat(train_x, train_y):
    train_x['click_mode'] = train_y

    train = train_x[train_x['req_time']<'2018-11-24']
    valid = train_x[train_x['req_time']>='2018-11-24']
    train = train.drop(['req_time'], axis=1)
    valid = valid.drop(['req_time'], axis=1)
    del train_x, train_y
    train_y = train[['click_mode']]
    train_x = train.drop('click_mode',axis=1)
    valid_y = valid[['click_mode']]
    valid_x = valid.drop('click_mode', axis=1)


    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']

    params = dict()
    params['learning_rate'] = 0.10
    params['depth'] = 5
    params['l2_leaf_reg'] = 4
    params['rsm'] = 1.0

    model = cat.CatBoostClassifier(
        iterations=10000,
        learning_rate=params['learning_rate'],
        depth=int(params['depth']),
        loss_function='MultiClass',
        use_best_model=True,
        eval_metric='MultiClass',
        l2_leaf_reg=params['l2_leaf_reg'],
        random_seed=2019,
        verbose=1,early_stopping_rounds=50,
        thread_count=2,
    )
    model.fit(train_x,train_y['click_mode'],early_stopping_rounds=50,eval_set=[(valid_x,valid_y)],verbose=1)
    val_pred = np.argmax(model.predict_proba(
        valid_x), axis=1)
    val_score = f1_score(valid_y, val_pred, average='weighted')
    print ('valid f1-score:' ,val_score)



def train_cat(train_x, train_y, test_x):
    train_x = train_x.drop(['req_time'], axis=1)
    test_x = test_x.drop(['req_time'], axis=1)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

    params = {
        'gamma': 0.2,
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 5,
        'eval_metric': 'merror',
        'seed': 2019,
        'missing': -999,
        'num_class': 12,
        'silent': 0,
        'lambda': 1,
        'alpha': 1,
        'nthread': 2,
    }
    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']
    scores = []
    result_proba = []
    for tr_idx, val_idx in kfold.split(train_x, train_y):
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
        params = dict()
        params['learning_rate'] = 0.10
        params['depth'] = 5
        params['l2_leaf_reg'] = 4
        params['rsm'] = 1.0

        model = cat.CatBoostClassifier(
            iterations=10000,
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            loss_function='MultiClass',
            use_best_model=True,
            eval_metric='MultiClass',
            l2_leaf_reg=params['l2_leaf_reg'],
            random_seed=2019,
            verbose=1, early_stopping_rounds=50,
            thread_count=4,
        )
        model.fit(train_x, train_y['click_mode'], early_stopping_rounds=50, eval_set=[(valid_x, valid_y)], verbose=1)
        val_pred = np.argmax(model.predict_proba(
            val_x), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        result_proba.append(model.predict_proba(
            test_x))
        scores.append(val_score)
    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    return pred_test


if __name__ == '__main__':
    if 'data.csv' in os.listdir('../pre_data'):
        data = pd.read_csv('../pre_data/data.csv')
        train_x, train_y, test_x, submit = split_train_test(data)

    else:
        train_x, train_y, test_x, submit = feature.get_train_test_feas_data()

    online = 0
    if online == 0:
        valid_cat(train_x, train_y)
    else:
        result_cat = train_cat(train_x, train_y, test_x)
        merge = 0
        if merge == 0:
            submit_result(submit, result_cat, 'cat')
        else:
            merge_result(submit, result_cat, 'cat')