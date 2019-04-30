
from core.feature import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

class manual_split:
    def split(self, X_data, cut_point=54):
        return self.split_sk(X_data)
        #return self.split_range( X_data, cut_point)

        #return self.split_group(X_data)


    def split_range(self,X_data,  cut_point):
        feature = get_feature()

        tmp = feature.loc[X_data.index]
        tmp = tmp.reset_index()


        res = [(tmp[(tmp.day>=0) & (tmp.day<=cut_point-1) ].index,
                 tmp[(tmp.day>=cut_point) & (tmp.day<=60) ].index)]
        return tqdm(res, f'split_range:{cut_point}')

    def split_sk(self, X_data):
        feature = get_feature()
        feature = feature.loc[X_data.index]

        feature = feature.reset_index()

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
        split_fold = folds.split(feature.values, feature.click_mode.values)

        return tqdm(split_fold, 'split_sk')


    def split_group(self,X_data,  begin_point=0):
        feature = get_feature()
        feature = feature.loc[X_data.index]

        feature = feature.reset_index()
        val = feature[(feature.day >= 54) & (feature.day <= 60)]
        train = feature.loc[(feature.day >= begin_point) & (feature.day < 54)]


        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
        split_fold = folds.split(train.values, train.click_mode.values)

        res = []
        for trn_inx, _ in tqdm((split_fold), 'Split group'):

            res.append((train.iloc[trn_inx].index, val.index))
        return tqdm(res, f'split_group:{begin_point},{len(val)}')


    def split_ratio(self,X_data,  cut_point):
        feature = get_feature()

        tmp = feature.loc[X_data.index]
        tmp = tmp.reset_index()

        ratio = get_feature_partition(cut_begin=cut_point, cut_end=60)
        ratio = ratio.click_mode / ratio.click_mode.min()

        df_list = []
        for day in tqdm(range(0, cut_point), 'resample base on day'):
            train = tmp.loc[tmp.day == day]
            gp = train.click_mode.value_counts()
            gp = gp.loc[gp.index >= 0].sort_index()
            base = gp.min()
            sample_count = round(ratio * base).astype(int)

            for i in tqdm(range(0, 12), 'resample base on ratio'):
                cnt = sample_count.loc[i]
                df_base = train.loc[train.click_mode == i]
                if cnt==0 or cnt > len(df_base):
                    logger.warning(f'cnt>len(df_base), {cnt}>{len(df_base)}')
                    cnt = min(cnt, len(df_base))
                tmp_df = df_base.sample(cnt)
                df_list.append(tmp_df)
        logger.debug(f'DF_list size: {len(df_list)}')
        new_df = pd.concat(df_list)
        logger.info(new_df.click_mode.value_counts().sort_index())
        return [(new_df.index, tmp[tmp.day>=cut_point].index)]


    @timed()
    def resample_train(begin=54, end=60):
        feature = get_feature()
        feature = feature.loc[feature.click_mode>=-1]
        gp = feature.click_mode.value_counts()
        gp = gp.loc[gp.index>=0].sort_index()
        base = gp.min()

        ratio = get_feature_partition(cut_begin=begin, cut_end=end)
        ratio = ratio.click_mode/ratio.click_mode.min()
        sample_count = round(ratio*base).astype(int)

        new_df = feature.loc[feature.click_mode==-1]

        for i in tqdm(range(0, 12), 'resample base on ratio'):
            cnt = sample_count.loc[i]
            tmp_df = feature.loc[feature.click_mode == i].sample(cnt)
            new_df = pd.concat([new_df, tmp_df])
        logger.info(new_df.click_mode.value_counts().sort_index())
        return new_df
