input_folder = './input/data_set_phase1'
plan_items = ['distance', 'eta', 'price', 'transport_mode']

val_cut_point = 60-7

hash_precision = [6]

type_dict ={
 'sid':'int',
 'pid':'str',

 'day':'int',
 'weekday':'int',
 'hour':'int',
 'weekend':'int',
 'o0':'float',
 'o1':'float',
 'd0':'float',
 'd1':'float',

 'click_mode':'str'
}