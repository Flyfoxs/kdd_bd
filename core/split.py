
from core.feature import get_query
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from file_cache.utils.util_log import logger
from tqdm import tqdm

class manual_split:


    @staticmethod
    def split_sk(X_data, n_splits=5):
        feature =  get_query()#.copy()
        feature = feature.loc[X_data.index.astype(int)]

        #feature = feature.dropna(how='any')

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)

        logger.info(f'split_sk:{feature.shape}')
        #check_exception(feature)
        split_fold = folds.split(feature.values, feature.click_mode.values.astype(int))

        return split_fold

    @staticmethod
    def split(X_data, cv, n_splits=5):
        if cv:
            return manual_split.split_sk(X_data, n_splits)
        else:
            return manual_split.split_range(X_data, 47)
            #return manual_split.split_range(X_data, 54)


    @staticmethod
    def split_range(X_data,  cut_point):
        feature = get_query()
        feature.index = feature.index.astype(int)
        tmp = feature.loc[X_data.index.astype(int)]

        res = [(tmp[(tmp.day>=0) & (tmp.day<=cut_point-1) ].index,
                 tmp[(tmp.day>=cut_point) & (tmp.day<=60) ].index)]
        return res
    #
    # @staticmethod
    # def split_group(X_data,  begin_point=0):
    #     feature = get_query()
    #     feature.index = feature.index.astype(int)
    #     feature = feature.loc[X_data.index.astype(int)]
    #
    #     #feature = feature.reset_index()
    #     val = feature[(feature.day >= 54) & (feature.day <= 60)]
    #     train = feature.loc[(feature.day >= begin_point) & (feature.day < 54)]
    #
    #
    #     folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    #     split_fold = folds.split(train.values, train.click_mode.values)
    #
    #     res = []
    #     for trn_inx, _ in tqdm((split_fold), 'Split group'):
    #
    #         res.append((train.iloc[trn_inx].index, val.index))
    #     return tqdm(res, f'split_group:{begin_point},{len(val)}')

    #
    # @staticmethod
    # def split_random(X_data):
    #     kf = KFold(n_splits=5,
    #                shuffle=True,
    #                random_state=2019).split(X_data)
    #     return kf

if __name__ == '__main__' :
    for i in range(5):
        pass