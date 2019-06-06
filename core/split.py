
from core.feature import get_feature
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from file_cache.utils.util_log import logger
from tqdm import tqdm

class manual_split:


    @staticmethod
    def split_sk(X_data):
        feature = get_feature().copy()
        feature = feature.loc[X_data.index]

        feature = feature.reset_index()

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

        logger.info(f'split_sk:{feature.shape}')
        #check_exception(feature)
        split_fold = folds.split(feature.values, feature.click_mode.values)

        return split_fold

    @staticmethod
    def split(X_data, cut_point=47):
        #return self.split_dummy_sid(X_data)
        return manual_split.split_sk(X_data)
        #return self.split_range( X_data, cut_point) #cut_point#54, train:(445340,), val:(54660,)

        #return self.split_group(X_data)

    # def split_dummy_sid(self, X_data):
    #     X_data = X_data.reset_index()
    #     split_list = []
    #     for i in range(5):
    #         train_sid, val_sid = get_split_sid(i)
    #
    #         train_index = X_data.loc[X_data.sid.isin(train_sid)]
    #         val_index   = X_data.loc[X_data.sid.isin(val_sid) & (X_data.en_label==0)]
    #         split_list.append((train_index.index.values, val_index.index.values))
    #     return  split_list
    #


    def split_range(self,X_data,  cut_point):
        feature = get_feature()

        tmp = feature.loc[X_data.index]
        tmp = tmp.reset_index()


        res = [(tmp[(tmp.day>=0) & (tmp.day<=cut_point-1) ].index,
                 tmp[(tmp.day>=cut_point) & (tmp.day<=60) ].index)]
        return res

    @staticmethod
    def split_group(X_data,  begin_point=0):
        feature = get_feature()
        feature = feature.loc[X_data.index]

        feature = feature.reset_index()
        val = feature[(feature.day >= 54) & (feature.day <= 60)]
        train = feature.loc[(feature.day >= begin_point) & (feature.day < 54)]


        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
        split_fold = folds.split(train.values, train.click_mode.values)

        res = []
        for trn_inx, _ in tqdm((split_fold), 'Split group'):

            res.append((train.iloc[trn_inx].index, val.index))
        return tqdm(res, f'split_group:{begin_point},{len(val)}')


    @staticmethod
    def split_random(X_data):
        kf = KFold(n_splits=5,
                   shuffle=True,
                   random_state=2019).split(X_data)
        return kf

if __name__ == '__main__' :
    for i in range(5):
        pass