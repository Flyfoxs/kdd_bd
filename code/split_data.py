import pandas as pd
import numpy as np



def merge_raw_data():
    tr_queries = pd.read_csv('../data_set_phase2/train_queries_phase2.csv')
    tr_plans = pd.read_csv('../data_set_phase2/train_plans_phase2.csv')

    tr_click = pd.read_csv('../data_set_phase2/train_clicks_phase2.csv',dtype={'click_mode':int})

    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)

    return tr_data


train = merge_raw_data()
print (train.shape)
# print (train['click_mode'].value_counts()/float(len(train)))
# print (result['recommend_mode'].value_counts()/float(len(result)))

data = pd.DataFrame()
for index,value in train.groupby(['click_mode']):
    if int(index) in [2,7,1,5]:
        value = value.sample(frac=1,random_state=2020)
        data = pd.concat([data,value],axis=0)
    else:
        data = pd.concat([data,value],axis=0)
data.to_csv('../data_set_phase2/split_data.csv',index=False)

