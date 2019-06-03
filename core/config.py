input_folder = './input/data_set_phase2'

disable_phase1 = True

plan_items = ['distance', 'eta', 'price', 'transport_mode']


plan_rank = ['price_rank', 'distance_rank', 'eta_rank']

plan_items_mini = ['distance', 'eta', 'price']

# enhance_model = {0:0.8, 3:0.8, 4:0.8, 6:0.8, 9:0.8 }
enhance_model = {}

val_cut_point = 60 - 7

hash_precision = [6]

type_dict = {
    'sid': 'str',
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

cate_cols = []

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


col_order = ['pid', 'weekday', 'hour', 'weekend', 'o0', 'o1', 'd0', 'd1', 'o_hash_6', 'd_hash_6', 'o_d_hash_6', '1_distance', '1_eta', '1_price', '1_transport_mode', '2_distance', '2_eta', '2_price', '2_transport_mode', '3_distance', '3_eta', '3_price', '3_transport_mode', '4_distance', '4_eta', '4_price', '4_transport_mode', '5_distance', '5_eta', '5_price', '5_transport_mode', '6_distance', '6_eta', '6_price', '6_transport_mode', '7_distance', '7_eta', '7_price', '7_transport_mode', '8_distance', '8_eta', '8_price', '8_transport_mode', '9_distance', '9_eta', '9_price', '9_transport_mode', '10_distance', '10_eta', '10_price', '10_transport_mode', '11_distance', '11_eta', '11_price', '11_transport_mode', '1_distance_max_p', '1_eta_max_p', '1_price_max_p', '10_distance_max_p', '10_eta_max_p', '10_price_max_p', '11_distance_max_p', '11_eta_max_p', '11_price_max_p', '2_distance_max_p', '2_eta_max_p', '2_price_max_p', '3_distance_max_p', '3_eta_max_p', '3_price_max_p', '4_distance_max_p', '4_eta_max_p', '4_price_max_p', '5_distance_max_p', '5_eta_max_p', '5_price_max_p', '6_distance_max_p', '6_eta_max_p', '6_price_max_p', '7_distance_max_p', '7_eta_max_p', '7_price_max_p', '8_distance_max_p', '8_eta_max_p', '8_price_max_p', '9_distance_max_p', '9_eta_max_p', '9_price_max_p', 'o_seq_0', 'o_seq_1', 'o_seq_2', 'o_seq_3', 'o_seq_4', 'o_seq_5', 'o_seq_6', 'glb_sugg_o_1', 'glb_sugg_o_2', 'glb_sugg_o_3', 'glb_sugg_o_4', 'glb_sugg_o_5', 'glb_sugg_o_6', 'glb_sugg_o_7', 'glb_sugg_o_8', 'glb_sugg_o_9', 'glb_sugg_o_10', 'glb_sugg_o_11', 'glb_day_appear_nunique_o', 'glb_count_appear_o', 'glb_sugg_o_1_per', 'glb_sugg_o_2_per', 'glb_sugg_o_3_per', 'glb_sugg_o_4_per', 'glb_sugg_o_5_per', 'glb_sugg_o_6_per', 'glb_sugg_o_7_per', 'glb_sugg_o_8_per', 'glb_sugg_o_9_per', 'glb_sugg_o_10_per', 'glb_sugg_o_11_per', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49', 'p50', 'p51', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p58', 'p59', 'p60', 'p61', 'p62', 'p63', 'p64', 'p65', 'profile_lda_0', 'profile_lda_1', 'profile_lda_2', 'profile_lda_3', 'profile_lda_4'
             ]


word2vec_list = ['01_distance_rank','01_eta_rank','01_price_rank','01_transport_mode','02_distance_rank','02_eta_rank','02_price','02_price_rank','02_transport_mode','03_distance_rank','03_eta_rank','03_price_rank','03_transport_mode','04_distance_rank','04_eta_rank','04_price_rank','04_transport_mode','05_distance_rank','05_eta_rank','05_price_rank','05_transport_mode','06_distance_rank','06_eta_rank','06_price_rank','06_transport_mode','07_distance_rank','07_eta_rank','07_price_rank','07_transport_mode','08_distance_rank','08_eta_rank','08_price_rank','08_transport_mode','09_distance_rank','09_eta_rank','09_price','09_price_rank','09_transport_mode','10_distance_rank','10_eta_rank','10_price_rank','10_transport_mode','11_distance_rank','11_eta_rank','11_price','11_price_rank','11_transport_mode',
                 'hour','weekday','weekend',
                 'o_seq_0','o_seq_1','o_seq_2','o_seq_3','o_seq_4','o_seq_5','o_seq_6',
                 'p0','p1','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p2','p20','p21','p22','p23','p24','p25','p26','p27','p28','p29','p3','p30','p31','p32','p33','p34','p35','p36','p37','p38','p39','p4','p40','p41','p42','p43','p44','p45','p46','p47','p48','p49','p5','p50','p51','p52','p53','p54','p55','p56','p57','p58','p59','p6','p60','p61','p62','p63','p64','p65','p7','p8','p9',
                 #'click_mode',
                 ]