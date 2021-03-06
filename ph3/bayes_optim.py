import sys

sys.path.append('./')

import numpy as np
import os

from bayes_opt import BayesianOptimization

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import warnings
warnings.filterwarnings('ignore')

def split_train_test(data):
    train_data = data[data['click_mode'].notnull()]
    test_data = data[data['click_mode'].isnull()]
    df_oof = train_data[['sid']].copy()
    df_submit = test_data[['sid']].copy()
    
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, df_oof, df_submit

def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True

def check_log():
    if os.path.exists('result_logs.json'):
        print('result_logs.json 文件存在')
        if os.path.exists('logs.json'):
            print('logs.json 文件存在，将logs.json写入result_logs.json')
            file = open( "logs.json", "r" )
            content = file.read()
            file.close()
            file_result = 'result_logs.json'
            with open(file_result, 'w+') as f:
                f.write(content)
            f.close()
            os.remove("logs.json")

        else:
            print('logs.json 文件不存在，无法追加内容！')
    else:
        print('result_logs.json 文件不存在')
        newfile = 'result_logs.json'
        f = open(newfile,'w')
        f.close()
        print('result_logs.json 文件创建成功！')

def lgbm_evaluate(**params):
    warnings.simplefilter('ignore')
    train_x, train_y, test_data, df_oof, df_submit = split_train_test(all_data)
    feature_name = [i for i in train_x.columns if i not in ['sid','click_mode','plan_time','req_time']]
    train_x = train_x[feature_name]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    
    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate': params['learning_rate'],
        'num_leaves': int(params['num_leaves']),
        'lambda_l1': params['lambda_l1'],
        'lambda_l2': int(params['lambda_l2']),
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'bagging_freq': int(params['bagging_freq']),
        'subsample':params['subsample'],
        'colsample_bytree':params['colsample_bytree'],
        'subsample_freq':1,
        'max_depth':-1,
        'nthread':6, #这里调整cpu核数
        'reg_alpha':params['reg_alpha'],
        'reg_lambda':int(params['reg_lambda']),
    }
    
    iter_n = 50
    
    scores = []
    for fold_, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y),1):
        print("fold: {}".format(fold_))
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]

        # cate features verify
        train_set = lgb.Dataset(tr_x, tr_y)
        val_set = lgb.Dataset(val_x, val_y)
        
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=iter_n, num_boost_round=40000, verbose_eval=iter_n, feval=eval_f)
        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        scores.append(val_score)

    return np.mean(scores)

"""
            lgb_model = lgb.LGBMClassifier(
                boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
                max_depth=-1, n_estimators=3000, objective='multiclass', num_classes=12,
                subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
                learning_rate=0.1, random_state=2019 + index, n_jobs=6, metric="None", importance_type='gain'
            )


"""

gbm_params = {
    'learning_rate': (0.1, 0.1),
    'num_leaves': (32, 128+36), ####
    'lambda_l1': (0.1, 0.1),
    'lambda_l2': (10, 10),
    'feature_fraction': (0.5, 0.5),
    'bagging_fraction': (0.5, 0.5),
    'bagging_freq': (3, 3),
    'subsample':(0.5, 0.7), #####
    'colsample_bytree':(0.5, 0.5),
    'reg_alpha':(0.1, 0.1),
    'reg_lambda':(5, 5),
}

from ph3.kdd_phase3_refactor import get_feature_all, get_feature_name


all_data = get_feature_all()

feature_name = get_feature_name(all_data)

all_data = all_data.loc[:,set(feature_name+['sid', 'click_mode'])]

all_data = all_data.sample(n=2000,random_state=123,axis=0) #这里做采样，验证逻辑

init_n = 20 #初始化迭代次数
iter_n = 200 #bayes opt迭代次数
check_log()
bo = BayesianOptimization(lgbm_evaluate, gbm_params)
logger = JSONLogger(path="logs.json")
bo.subscribe(Events.OPTMIZATION_STEP, logger)

# Results will be saved in logs.json
load_logs(bo, logs=["result_logs.json"])

bo.maximize(init_points = init_n, n_iter = iter_n)

best_params = bo.max['params']
best_target = bo.max['target']
print(best_target, best_params)



""""
nohup python -u ph3/bayes_optim.py > optim.log 2>&1  & 
"""






