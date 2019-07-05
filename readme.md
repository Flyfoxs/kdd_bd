# 配置数据文件存放目录
    rm -rf data
    ln -s ~/mnt/data/data_set_phase2 ./data

# 运行方式
nohup python -u metrics_collector.py >> metrics.log 2>&1 &