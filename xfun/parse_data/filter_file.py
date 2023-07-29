# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     filter_file
   Description :
   Author :       ASUS
   date：          2023-05-25
-------------------------------------------------
   Change Activity:
                   2023-05-25:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import re
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from utils.params_config import ConfPath, ConfVars, ConstVars


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


class ParseLand(ConfParams):
    def __init__(self):
        ConfParams.__init__(self)

    def filter_norm_data(self, src_path):
        """
        1. filter cex by: 1C 4.2V  5.0AH
        2.OUTPUT:
           record_file.xlsx
            path -filename -sys_name(batch)  sysyid(batchid)

        """
        file_record = {}
        for fold_cond in os.listdir(src_path):
            fold_path = os.path.join(src_path, fold_cond)
            for sub_fold in os.listdir(fold_path):
                # if (('5.0' in sub_fold) and ('4.2' in sub_fold)) and ('标准' in sub_fold):
                if (('5.0' in sub_fold) and ('4.2' in sub_fold)) \
                        and ('1C' not in sub_fold):
                    # and ('非车电' not in sub_fold)
                    file_fold = os.path.join(fold_path, sub_fold)
                    for file in os.listdir(file_fold):
                        if not file.endswith('.cex'):
                            continue
                        file_name = os.path.join(file_fold, file)
                        file_stats = os.stat(file_name).st_size / 1024 / 1024
                        if not file_stats > 3.8:
                            continue
                        file_record[file_name] = [file.split('.')[0], sub_fold, sub_fold.split('-')[0]]

        df_record = pd.DataFrame.from_dict(file_record, orient='index').reset_index()
        df_record.columns = ['path', 'filename', 'sys_name', 'sysid']
        df_record.to_excel('record_file.xlsx', sheet_name='record', index=False)

    def cp_file(self, dest_fold):
        """
        cp filtered file to destfold
        """
        df_record = pd.read_excel('record_file.xlsx', sheet_name='record')
        rows = range(0, df_record.shape[0])
        for k in rows:
            src_file, filename, _, _ = df_record.iloc[k, :]
            dest_file = os.path.join(dest_fold, filename + '.cex')
            shutil.copy(src_file, dest_file)
        print('move done')

    def split2sysfold(self, asm_xlsx_fold, asm_bysys_fold):
        """
        1. export  cex to xlsx in messy order
        2. move xlsx  into batch-sysyid fold
        """

        df_record = pd.read_excel('record_file.xlsx', sheet_name='record')
        for file in os.listdir(asm_xlsx_fold):
            if not file.endswith('.xls'):
                continue
            index_file = file.split('.')[0]
            sys_id_df_row = df_record[df_record['filename'] == index_file]
            if sys_id_df_row.shape[0] == 0:
                continue
            sys_id = sys_id_df_row['sysid'].values[0]

            sys_fold = os.path.join(asm_bysys_fold, sys_id)
            if not os.path.exists(sys_fold):
                os.makedirs(sys_fold)
            xls_path_filename = os.path.join(asm_xlsx_fold, file)
            sysid_path_filename = os.path.join(sys_fold, file)
            shutil.move(xls_path_filename, sysid_path_filename)
        print('move2sysID done')

    def plt_asm(self, src_data_path):
        """
        drop:  '22D16TA', '22C27TA', '22C24TA'
        plot by： trend、capacity-rentention

                # if sys_fold in ('21L31TA', '22A02TA', '22A29TA', '22B01TA',
                #                 '22B02TA', '22B12TA', '22B19TA', '22B23TA', '22D25TA',
                #                 '22C06TA', '22H22TA', '22D18TA', '22C20TA', '22D26TA', ):

        """
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
        color_set = cm.rainbow(np.linspace(0, 1, 1500))
        k = 0

        for fold in os.listdir(src_data_path):
            # if not (fold.startswith('class2') or fold.startswith('class1') or fold.startswith('class4')):
            # if (fold.startswith('tmp') and (not fold.startswith('class4'))):
            # if not (fold.startswith('class2')):
            # if not fold.startswith('test'):
            #     continue
            # print(fold)

            sub_class_path = os.path.join(src_data_path, fold)
            if not os.path.isdir(sub_class_path):
                continue
            try:
                for sys_fold in os.listdir(sub_class_path):
                    if sys_fold.endswith('lower'):
                        continue
                    sys_path = os.path.join(sub_class_path, sys_fold)
                    icolor = color_set[k]

                    for file in os.listdir(sys_path):
                        if not file.endswith('.xls'):
                            continue
                        file_path = os.path.join(sys_path, file)
                        df = pd.read_excel(file_path, sheet_name='循环', usecols=['放电容量/mAh'])
                        # df = df.rolling(window=8, min_periods=3).mean().values[5:-5]
                        df = df.rolling(window=8, min_periods=3).mean()
                        label = sys_fold + ':' + file.split('.')[0]
                        # df_y = df.iloc[200, :].values[0]
                        # if df_y < 4600:
                        #     print(label)
                        #     ax.plot(df, color=icolor, linewidth=2, label=label)
                        # else:
                        #     continue
                        if (sys_fold.endswith('outlier') and sys_fold.startswith('G24')):
                            ax.plot(df, color='k', label=label, linewidth=4)
                        elif (sys_fold.endswith('outlier') and sys_fold.startswith('G25')):
                            ax.plot(df, color='m', label=label, linewidth=4)
                        elif (sys_fold.endswith('outlier') and sys_fold.startswith('I26')):
                            ax.plot(df, color='hotpink', label=label, linewidth=4)
                        elif (sys_fold.endswith('outlier') and sys_fold.startswith('G23')):
                            ax.plot(df, color='y', label=label, linewidth=4)
                        elif sys_fold.startswith('SC9'):
                            ax.plot(df, color='r', label=label)
                        elif sys_fold.startswith('G23'):
                            ax.plot(df, color='orange', label=label)
                        elif (sys_fold.endswith('outlier') and sys_fold.startswith('SC1')):
                            ax.plot(df, color='pink', label=label, linewidth=4)
                        elif (sys_fold.endswith('lower') and sys_fold.startswith('G22')):
                            ax.plot(df, color='orangered', label=label, linewidth=4)
                        else:
                            ax.plot(df, color=icolor, linewidth=2, label=label)
                    k = k + 30
            except Exception as err:
                print('cex read errors:{}'.format(err))
        xlabels = range(0, 2500, 100)
        ylabels = range(2500, 5000, 150)
        ax.set_xticks(xlabels)
        ax.set_yticks(ylabels)
        ax.set_ylim((2500, 5000))
        ax.set_xticklabels(xlabels, fontsize=12, rotation=-60)
        ax.legend(fontsize=6, loc='best')
        ax.grid()
        plt.savefig(os.path.join('sys_clf_fold', 'bysysID_asm.png'), dpi=350)
        # plt.savefig(os.path.join('sys_clf_fold', 'bysysID_asm-t.png'), dpi=350)

    def rebuild_cex(self, asm_cex_cyl1000, src_sys_xlsx_path, src_cex_sel_path, src_xlsx_selected):

        for sys_fold in os.listdir(src_sys_xlsx_path):
            if sys_fold.endswith('lower'):
                continue
            sys_path = os.path.join(src_sys_xlsx_path, sys_fold)
            for file in os.listdir(sys_path):
                if not file.endswith('.xls'):
                    continue
                cex_nname = file.split('.')[0] + '.cex'
                cex_file_name = os.path.join(asm_cex_cyl1000, cex_nname)
                dest_cex_sel_fold = os.path.join(src_cex_sel_path, sys_fold)
                dest_xlsx_sel_fold = os.path.join(src_xlsx_selected, sys_fold)
                if not os.path.exists(dest_cex_sel_fold):
                    os.makedirs(dest_cex_sel_fold)
                if not os.path.exists(dest_xlsx_sel_fold):
                    os.makedirs(dest_xlsx_sel_fold)
                dest_cex_sel_file = os.path.join(dest_cex_sel_fold, cex_nname)
                shutil.copy(cex_file_name, dest_cex_sel_file)
        print('move selected cex done')


def run_parse_land():
    src_path = r'D:\郑州循环数据-汇总\三部\3部常温'
    dest_fold = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\src_cex-class4'
    asm_xlsx_fold = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\class4-xlsx'
    asm_bysys_fold = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\class4-bysys'
    asm_xlsx_cyl = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\class4-xlsx-循环层'
    asm_bysys_cylfold = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\class4-bysys-cyl'
    src_data_path = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data'

    asm_cex_cyl1000 = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\sasm_cex-cyl1000'
    src_sysxlsx_sel_path = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\class4-bysys-cyl'
    src_cex_sel_path = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\src_cex_selected'
    src_xlsx_selected = r'E:\code_git\郑州品质_跳水_数据\0.5C\src_data\src_xlsx_selected'
    pl = ParseLand()
    # 1.filter cex by: 1C 4.2V  5.0AH
    # pl.filter_norm_data(src_path)
    # 2. cp cex to dest_fold
    # pl.cp_file(dest_fold)
    # export cex to xlsx by manually
    # 3.  move xlsx  into batch-sysyid fold
    # pl.split2sysfold(asm_xlsx_fold, asm_bysys_fold)
    # pl.split2sysfold(asm_xlsx_cyl, asm_bysys_cylfold)
    # 4.
    src_data_path = r'E:\code_git\郑州品质_跳水_数据\0.5C_batch'
    pl.plt_asm(src_data_path)
    # pl.rebuild_cex(asm_cex_cyl1000, src_sysxlsx_sel_path, src_cex_sel_path, src_xlsx_selected)


if __name__ == "__main__":
    run_parse_land()
