import sys
sys.path.append('./')

from glob import glob
import lightgbm as lgb
from file_cache.utils.util_log import logger, timed
import numpy as np

from ph3.kdd_phase3_refactor import get_feature_all, get_feature_name, get_predict_feature
import pandas as pd
from tqdm import tqdm





@timed()
def predict(*args, **kwargs):

    X_test = get_predict_feature()

    recommend_mode = np.zeros((len(X_test), 12))
    for model_file in glob('./model/*.pkl'):
        logger.info(f'model_file:{model_file}')
        from sklearn.externals import joblib
        # load model
        lgb_model = joblib.load(model_file)
        y_test = lgb_model.predict_proba(X_test)

        recommend_mode += y_test

    sub = pd.DataFrame(recommend_mode/5 , index=X_test.index)

    sub = adjust(sub)

    sub.index.name='sid'
    sub['recommend_mode'] = sub.idxmax(axis=1)

    sub_file = f'./result/predict.csv'
    sub[['recommend_mode']].to_csv(sub_file)
    logger.debug(f'Predict file save to {sub_file}')
    return sub_file


def adjust(sub):
    from kdd_train import OptimizedRounder as opt
    opt = opt()
    coef = opt.initial_coef
    for i in range(12):
        sub.loc[:, i] = sub.loc[:, i]* coef[i]
    return sub


if __name__ == '__main__':
    import fire
    fire.Fire()

    """"
    
    nohup python -u kdd_predict.py predict > predict.log 2>&1  &
    """