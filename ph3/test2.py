import pandas as pd

#/home/jovyan/mnt/felix/kdd_bd_new/cache/get_feature_all==.pickle
def get_feature_all():
    tmp = pd.read_pickle('./cache/get_feature_all==.pickle')
    print(tmp.shape)
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


@timed()
def convert_df(df:pd.DataFrame):
    partition_num = 5
    for i in tqdm(range(partition_num)):
        with timed_bolck(f'Convert#{i}'):
            todo = [index for index in range(df.shape[1]) if index % partition_num == i]
            df.iloc[:, todo] = df.iloc[:, todo].astype(np.float32)
            gc.collect()
    for name, types in df.dtypes.iteritems():
        print(name,types)

    return df


@timed()
def train_base(feature_cnt=9999):
    all_data = get_feature_all().fillna(0)  # .sample(frac=0.5)
    # Define F1 Train
    feature_name = get_feature_name(all_data)[:feature_cnt]
    logger.debug(f'Final Train feature#{len(feature_name)}')
    # CV TRAIN

    logger.info('Begin values')
    tr_index = ~all_data['click_mode'].isnull()
    X_train = all_data[tr_index][list(set(feature_name))].reset_index(drop=True)#.values
    y = all_data[tr_index]['click_mode'].reset_index(drop=True)#.values
    X_test = all_data[~tr_index][list(set(feature_name))].reset_index(drop=True)#.values
    print(X_train.shape, X_test.shape)
    logger.info('End values')
    final_pred = []
    cv_score = []
    cv_model = []
    del all_data

    X_train = convert_df(X_train)

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


        # logger.info(X_train.dtypes.value_counts)
        # with timed_bolck('Begin Convert float'):
        #     for col in tqdm(X_train.columns):
        #         gc.collect()
        #         # if 'float' not in X_train.loc[:,col].dtype.name:
        #         X_train.loc[:,col] = X_train.loc[:,col].astype(float)



        logger.info('Begin Split')
        train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        eval_set = [(test_x, test_y)]
        logger.info('End Split')
        logger.info(f'Begin Manually convert')

        # gc.collect()
        train_x = train_x.values
        # logger.info('Convert to values')

        gc.collect()
        #train_x = np.array(train_x.values.reshape(train_x.size), dtype=np.float32, copy=False)
        logger.info(f'Value convert,{train_x.shape}, <<<crash point>>>')
        lgb_model.fit(train_x
                      , train_y.values, eval_set=eval_set, verbose=10,
                      early_stopping_rounds=30, eval_metric=f1_macro )
        logger.info('End Fit, <<<crash point>>>')
        #logger.info(f'End Train#{index}, best_iter:{lgb_model.best_iteration_}')

if __name__ == '__main__':
    import fire
    fire.Fire()
