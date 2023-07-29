#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logger
   Description :
   Author :       SV00240663
   date：          2021/8/21
-------------------------------------------------
   Change Activity:
                   2021/8/21:
-------------------------------------------------
"""
__author__ = 'SV00240663'

import os
import datetime
import logging


def log_run():
    current_path = os.path.dirname(__file__)
    log_path = os.path.join(current_path, '..\\log\\')
    # 创建一个日志器logger并设置其日志级别为DEBUG
    logger = logging.getLogger('simple_logger')
    logging.basicConfig(
        level=logging.INFO,
        # filename="BlogNetease.log",
        filemode='wa',
    )
    dt = datetime.datetime.now().strftime(format('%Y-%m-%d-%H-%M-%S'))
    handler = logging.FileHandler(log_path + dt + '_TSC.log', mode='a')
    # 创建一个格式器formatter并将其添加到处理器handler
    formatter = logging.Formatter(
        "%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    # 为日志器logger添加上面创建的处理器handler
    logger.addHandler(handler)
    return logger


logger = log_run()
