import pandas as pd


def get_feature_all():
    tmp = pd.read_pickle('/home/jovyan/mnt/felix/kdd_bd_new/cache/get_feature_all==.pickle')
    return tmp


from tqdm import tqdm
from file_cache.utils.util_log import timed, timed_bolck, logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
import gc


def get_feature_name(df):
    feature_name = [i for i in df.columns if i not in ['sid', 'click_mode', 'plan_time', 'req_time', 'label', 'type_']]
    return feature_name


def f1_macro(labels, preds):
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score, True



def train_base(feature_cnt=9999):
    all_data = get_feature_all()  # .sample(frac=0.5)
    # Define F1 Train
    feature_name = get_feature_name(all_data)[:feature_cnt]
    logger.debug(f'Final Train feature#{len(feature_name)}')
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
        gc.collect()
        # with timed_bolck(f'Folder#{index}, feature:{len(feature_name)}'):
        lgb_model = lgb.LGBMClassifier(
            boosting_type="gbdt", num_leaves=128, reg_alpha=0.1, reg_lambda=10,
            max_depth=-1, n_estimators=3000, objective='multiclass', num_classes=12,
            subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
            learning_rate=0.1, random_state=2019 + index, n_jobs=10, metric="None", importance_type='gain'
        )
        train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[
            test_index], y.iloc[train_index], y.iloc[test_index]
        eval_set = [(test_x[feature_name], test_y)]
        logger.info(f'Begin Train#{index}, feature:{len(feature_name)}, Size:{train_x[feature_name].shape}')
        lgb_model.fit(train_x[feature_name], train_y, eval_set=eval_set, verbose=10, early_stopping_rounds=30,
                      eval_metric=f1_macro)
        logger.info(f'End Train#{index}, best_iter:{lgb_model.best_iteration_}')


train_base()