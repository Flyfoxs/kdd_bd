#!/usr/bin/python
#-*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
File: metrics_collector.py 
Author: kddcup2019(kddcup2019-support@baidu.com) 
Date: 2019-06-17 11:26
Description: KDD Efficiency Metrics Collector  
"""

"""
 RULE:
1) Train part:
    a) Use kdd_train.py as the target module, train() as the main function;
    b) In this part, we record the time from loading original data to obtaining a model as total time cost;
    c) Memory utilization will be collected every 10 seconds, and get the average utilization.
2) Predict part:
    a) Use kdd_predict.py as the target module, predict() as the main function;
    b) In this part, we record the time from loading original data to obtaining a model as total time cost;
    c) Memory utilization will be collected every 10 seconds, and get the average utilization.
3) Original data store in ~/data; model file store in ~/model; predict result file store in ~/result
4) Any other tmp file, can be stored in other allowed directory; 
5) We will run part 1) and 2), to collect the metrics
6) Besides, we will review all the source code to make sure there is no cheating.
"""


import time
import json
import os
from threading import Thread

import psutil
try:
    import nvidia_smi
except:
    print('Import nvidia_smi error')


class MemoryDetector(Thread):
    """memory detector thread"""
    def __init__(self, stop_detect=False, interval=1):
        Thread.__init__(self)
        self.stop_detect = stop_detect 
        self.interval = interval
        self.memory_info = dict()

    def _memory_usage(self):
        cur_process = psutil.Process(os.getpid())
        pro_list = [cur_process] + cur_process.children(recursive=True)
        memory_usage = 0.0
        for pro in pro_list:
            memory_usage += pro.memory_percent()
        return memory_usage

    def _gpu_usage(self):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # card id 0 hardcoded here, we can also call to get all available card ids to iterate
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        return res.memory # res.gpu is not necessary by now

    def run(self):
        """detect metrics, we need two information: 1) memory 2) gpu memory
           the time we detect may ${self.interval} longer than train/predict period  
        """
        counter = 0
        memory = 0.0
        gpu = 0.0
        while not self.stop_detect:
            counter += 1
            memory += self._memory_usage()
            gpu += self._gpu_usage()
            time.sleep(self.interval)
        print(time.time(), "stop detect")
        self.memory_info = {"counter": counter, "memory": memory, "gpu": gpu}
        print(time.time(), self.memory_info)
    
    def stop(self):
        """stop detector work"""
        self.stop_detect = True

    def get_memory_info(self):
        """get the result info of detector"""
        return self.memory_info


def detector(func):
    """decorator for efficency detector"""
    def wrapper(self):
        """wrapper"""
        m_detector = MemoryDetector()
        m_detector.start()

        s_time = time.time()
        func(self)
        e_time = time.time()

        m_detector.stop()
        m_detector.join()

        metrics = dict()
        metrics["cost"] = e_time - s_time
        metrics["memory"] = m_detector.get_memory_info()
        return metrics
    return wrapper


class Collector(object):
    """metrics collector"""

    @detector
    def train(self):
        """train metrics collector"""
        import kdd_train
        kdd_train.train()

    @detector
    def predict(self):
        """predict metrics collector"""
        test_1 = "~/data/test_data_1"
        # if test_data is not none, use it as test data; 
        # otherwise use original test data as default
        import kdd_predict
        kdd_predict.predict(test_data=test_1)
     

if __name__ == "__main__":
    """main function"""
    result = dict()
    collector = Collector()
    result["train"] = collector.train()
    result["predict"] = collector.predict()
    print(json.dumps(result))

"""
    nohup python -u metrics_collector.py >> metrics_05.log 2>&1 &
"""