input_folder = './input/data_set_phase1'

plan_items = ['distance', 'eta', 'price', 'transport_mode']

plan_items_mini = ['distance', 'eta', 'price']

# enhance_model = {0:0.8, 3:0.8, 4:0.8, 6:0.8, 9:0.8 }
enhance_model = {}

val_cut_point = 60 - 7

hash_precision = [6]

type_dict = {
    'sid': 'int',
    'pid': 'str',

    'day': 'int',
    'weekday': 'int',
    'hour': 'int',
    'weekend': 'int',
    'o0': 'float',
    'o1': 'float',
    'd0': 'float',
    'd1': 'float',

    'click_mode': 'str'
}

cate_cols = ['weekend', 'weekday']

good_col = [
    '6_distance_max_p', '1_price_max_p', 'profile_lda_1', '6_eta_max_p', '4_price', '2_eta', 'glb_sugg_o_6_per',
    'glb_sugg_o_2_per', '3_distance_max_p', '5_eta_max_p', '2_price_max_p', 'glb_sugg_o_11', 'hour', '6_distance',
    'o_seq_3', 'glb_sugg_o_7', 'profile_lda_3', 'glb_sugg_o_4', 'glb_sugg_o_10_per', '4_distance', '10_price_max_p',
    '2_eta_max_p', 'glb_sugg_o_3', '10_eta_max_p', 'profile_lda_0', 'glb_sugg_o_9', 'o_seq_4', '1_distance',
    '9_eta_max_p', 'o_hash_6', 'o0', '7_eta', 'glb_sugg_o_5_per', 'glb_count_appear_o', 'glb_sugg_o_10', 'glb_sugg_o_8',
    'profile_lda_2', 'glb_sugg_o_3_per', '3_eta_max_p', 'glb_sugg_o_7_per', '4_distance_max_p', '7_price_max_p', 'pid',
    'glb_sugg_o_11_per', 'profile_lda_4', '5_distance', 'd0', 'o1', '1_eta', '7_distance_max_p', '9_distance',
    '1_eta_max_p', 'glb_sugg_o_6', '9_eta', '3_eta', 'glb_sugg_o_8_per', 'glb_sugg_o_1', '3_distance', 'd1',
    '9_price_max_p', '2_distance', 'glb_sugg_o_5', '7_eta_max_p', 'glb_sugg_o_9_per', 'glb_sugg_o_2',
    'glb_sugg_o_1_per', '9_distance_max_p', 'glb_sugg_o_4_per', 'o_seq_2', 'o_seq_0'
].extend(cate_cols)
