

# Features


## commit -3

����һЩgeohash ������������6953 ��������6968
������0.5

## commit -2

Χ��rank����һЩ����

������ 0.9 ����6916

0.8 + rank ��������6953


## commit 1 

������һЩ���� 

1.plansԭʼ��ȡmode����mode����
  
2.plansԭʼ���ݳ�ȡspeed=distance/eta,��speed����

3.����ԭʼ����

4.Ŀ�ĵ�������������پ������

5.Ŀ�ĵ�������ؽǶȼ���

6.ͳ���Ƽ���plans����

7.ͳ��plans����price����plans��������

8.plans��mode-distance����

9.plans��mode-eta����

10.plans��mode-price����(���øþ�����ʶ����Щmode����Ҫprice����Щ����Ҫprice)   mode 3 5 6Ϊû��price��mode

11.plans�е�max min mean price

12.plans�е�max min mean eta

13.plans�е�max min mean distance

14.����λ��������ࡣ

15.mode 3 5 6��plans�г��ֵĸ���

16.profile�û�����Ϊw2v d2v tf-idf

17.transport����mode tf-idf svd

18.time week day month

19.profile svd

20.mean max min mode

21.transport_mode����ͬ��mode����ͬ����  ��һ��mode���ֺ��Ƿ��ٳ��֣��ٳ��ִ���

22.kmeans����ָ����

23.����ģʽ�ֽ�transport_mode


## �Ƚόŵ�
 
��ά���� mode_price  mode_distance  mode_eta

transport_mode_svd_fea_2 ǿ��

recommand_0_transport_mode ǿ��

price_svd_fea_1 ǿ��

# How to use:

1. put all "xxx.csv" files into "data" dir.

2. run "python gbdt.py" in "code" dir.

3. you can get the result file in "submit" dir.

# Requirement

numpy - 1.16.2

pandas - 0.22.0

lightgbm - 2.1.0

sklearn - 0.19.1

tqdm - 4.31.3

