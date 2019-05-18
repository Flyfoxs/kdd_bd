import pandas as pd
from time import strftime,gmtime
import numpy as np
model_1 = pd.read_csv('../merge/.csv')
model_2 = pd.read_csv('../merge/.csv')


result = model_1[['sid']]
model_1 = model_1.drop(['sid'],axis=1)
model_2 = model_2.drop(['sid'],axis=1)

now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
result['recommend_mode'] = np.argmax(model_1.values *0.7 + model_2 *0.3 ,axis=1)

result.to_csv(
    '../merge/merge_result_{}.csv'.format(now_time), index=False)
