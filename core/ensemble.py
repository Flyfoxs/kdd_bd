import pandas as pd

file_list = [

    './output/stacking/St_True_2000000_503_0.66988_0596_1354.h5',
    './output/stacking/St_True_2000000_503_0.66992_1043_1331.h5',
    './output/stacking/St_True_2000000_503_0.66967_0746_1457.h5',
    './output/stacking/C_True_2000000_503_0.66954_0653_1511.h5',
]

train = None
test = None
for file in  file_list:
    tmp_train = pd.read_hdf(file, 'train')
    tmp_train.click_mode = tmp_train.click_mode.astype(int)
    train = tmp_train if train is None else  train + tmp_train

    tmp_test = pd.read_hdf(file, 'test')
    # tmp_test.click_mode = tmp_test.click_mode.astype(int)
    test = tmp_test if test is None else test + tmp_test

train = train / 4
test = test / 4

train.to_hdf(f'./output/stacking/avg.h5', 'train')
test.to_hdf(f'./output/stacking/avg.h5', 'test')