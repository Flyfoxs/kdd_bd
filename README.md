

# Features


## commit -3

做了一些geohash 特征。分数从6953 提升到了6968
采样比0.5

## commit -2

围绕rank做了一些特征

采样比 0.9 分数6916

0.8 + rank 特征到了6953


## commit 1 

做过的一些特征 

1.plans原始抽取mode，做mode矩阵
  
2.plans原始数据抽取speed=distance/eta,做speed矩阵

3.保留原始特征

4.目的地与出发地曼哈顿距离计算

5.目的地与出发地角度计算

6.统计推荐的plans数量

7.统计plans中有price（无plans）的数量

8.plans中mode-distance矩阵

9.plans中mode-eta矩阵

10.plans中mode-price矩阵(利用该矩阵来识别哪些mode是需要price，哪些不需要price)   mode 3 5 6为没有price的mode

11.plans中的max min mean price

12.plans中的max min mean eta

13.plans中的max min mean distance

14.地理位置坐标聚类。

15.mode 3 5 6在plans中出现的个数

16.profile用户描述为w2v d2v tf-idf

17.transport——mode tf-idf svd

18.time week day month

19.profile svd

20.mean max min mode

21.transport_mode中相同的mode和相同个数  第一个mode出现后是否再出现，再出现次数

22.kmeans聚类分割地区

23.主题模式分解transport_mode


## 比较屌的
 
三维矩阵 mode_price  mode_distance  mode_eta

transport_mode_svd_fea_2 强特

recommand_0_transport_mode 强特

price_svd_fea_1 强特

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

