# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     code_chk.py
   Description :
   Author :       ASUS
   date：          2023/4/17
-------------------------------------------------
   Change Activity:
                   2023/4/17:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os

import pycodestyle

path = r'E:\code_git\cleancode\CLP_LFP\xfun\train_model'
fchecker = pycodestyle.Checker(os.path.join(path, 'models_calcu.py'), show_source=True)
file_errors = fchecker.check_all()
print("Found %s errors (and warnings)" % file_errors)
