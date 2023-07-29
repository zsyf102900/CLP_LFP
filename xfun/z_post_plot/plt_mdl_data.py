"""
-------------------------------------------------
   File Name：    $ {NAME}.py 
   Description :
   Author :       
   date：         2023/3/3/10:39
-------------------------------------------------
   Change Activity:
                     2023/3/3/10:39
-------------------------------------------------
"""
import os
import pickle
import numpy as np
import pandas as pd
import autoencoder
import tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import entropy, norm, spearmanr
from utils.params_config import ConfPath, ConfVars, ConstVars

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tk = tkinter.Tk()
width = tk.winfo_screenwidth() / 100
height = tk.winfo_screenheight() / 100


class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


class PostVisual(ConfParams):

    def __init__(self, mdl_name, pred_mode, x_range, y_range):
        """
        /data6/test/CE01_ProbeCheck/cap_pkl_CV04/
        """
        ConfParams.__init__(self)
        self.mdl_name = mdl_name
        self.pred_mode = pred_mode
        self.df_tr = pd.read_excel(os.path.join(self.ts_avg_fatures_path, 'TRAIN_mdldata.xlsx'), sheet_name='cyl_data')
        with open(os.path.join(os.path.join(self.ts_avg_fatures_path, 'PRED_cap_join_mdldata.pkl')), 'rb') as rf:
            df_pred = pickle.load(rf)
        # self.df_pred = pd.concat(df_pred, axis=0)
        self.df_corr_spman = pd.read_csv(os.path.join(self.pkl_path_mdl, 'df_corr_spman.csv'), sep=',', index_col=0)
        self.df_imp = pd.read_csv(os.path.join(self.pkl_path_mdl, 'df_importance.csv'), sep=',', index_col=0)
        self.dpi = 800
        self.xtick_step = 100
        self.x_range = x_range
        self.y_range = y_range
        maxcolor = 1024
        stepcolor = 3
        self.color_rbow = plt.cm.rainbow(np.linspace(0, stepcolor, maxcolor))
        self.color_vtd = plt.cm.prism(np.linspace(0, stepcolor, maxcolor))
        self.color_sq = plt.cm.jet(np.linspace(0, stepcolor, maxcolor))

    def plt_dist(self):
        """
        scatter, corr:heatmap
        export data
        :param df_src:
        :param df_corr_spman:
        :return:
        """

        features = self.df_imp['feature_names'].tolist() + [self.y_label]
        for col in features:
            try:
                fig1, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
                sns.distplot(self.df_tr[col], rug=True, bins=10, ax=ax1, color='m', label='TRAIN')
                sns.distplot(self.df_pred[col], rug=True, bins=10, ax=ax1, color='c', label='PRED')

                ax1.set_xlabel(xlabel='{} PDF'.format(col), fontsize=12)
                ax1.set_ylabel(ylabel='{} PDF'.format(col), fontsize=12)
                ax1.set_title('{} PDF'.format(col), fontsize=16)
                ax1.legend()
                path_dist = os.path.join(self.path_png_plt, 'dist_PDF_png')
                if not os.path.exists(path_dist):
                    os.makedirs(path_dist)
                if '/' in col:
                    col = col.replace('/', '-')
                plt.savefig(os.path.join(path_dist, col + '_norm_dist.png'), dpi=self.dpi)
            except Exception as err:
                print('features:{} not in '.format(col))
            # plt.show()
        plt.close()

    def plt_cmp_all(self):
        features = ['END_CAPACITY', 'DISCHG_ENDENERGY', 'DISCHG_INITVOL', 'DISCHG_AVGVOL',
                    'CHG_ENDCAPACITY', 'CHG_ENDENERGY', 'CHG_INITVOL', 'CHG_AVGVOL',
                    'STAT_ENDVOL', 'DELTA_STATVOL',
                    'MEAN_CHGVOL', 'MEAN_DISCHGVOL', 'RV_VOLTAGE', 'SV_VOLTAGE',
                    'Q_EPSILON', 'CC_EPSILON', 'CYCLE_INT',
                    'fst_peak', 'sec_peak',
                    'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23'
                    ]

        for col in features:
            # fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

            fig1, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
            k = 0
            for fold_tr_pred in os.listdir(self.data_join_path):
                tr_path = os.path.join(self.data_join_path, fold_tr_pred)
                for sysfold in os.listdir(tr_path):
                    sysfold_path = os.path.join(tr_path, sysfold)
                    for file_fold in os.listdir(sysfold_path):
                        if '-E2' in file_fold:
                            continue
                        file_fold_path = os.path.join(sysfold_path, file_fold)
                        for file in os.listdir(file_fold_path):
                            if not file.endswith('summary.xlsx'):
                                continue
                            file_path = os.path.join(file_fold_path, file)
                            df_asm = pd.read_excel(file_path, sheet_name='cyl_data')
                            df_asm = df_asm.set_index(self.drop_cols).iloc[self.dqdv_initcycle:, :]
                            try:
                                ax1.plot(df_asm[col].rolling(window=10, min_periods=3).mean().values,
                                         c=self.color_rbow[k], label=sysfold + ':' + file_fold)
                                # ax2.plot(df_asm[col].diff().values, c=self.color_rbow[k], label=sysfold + ':' + file_fold)
                                k = k + 10
                            except Exception as err:
                                print('features:{} not in.err:{} '.format(col, err))
            path_features_sv = os.path.join(self.path_png_plt, 'features_cmp_fold')
            if not os.path.exists(path_features_sv):
                os.makedirs(path_features_sv)
            if '/' in col:
                col = col.replace('/', '-')
            ax1.set_title('{} '.format(col), fontsize=16)
            ax1.legend()
            plt.savefig(os.path.join(path_features_sv, col + '_.png'), dpi=self.dpi)
            plt.close()

    def plt_dqdv_diff(self):
        # features = ['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23']
        features = [
            'CHG_AVGVOL',
            'MEAN_CHGVOL', 'CHG_INITVOL',
            'DELTA_STATVOL', 'STAT_ENDVOL', 'RV_VOLTAGE', 'sec_peak', 'sec_peak_x', 'area1_q23'
        ]

        normal_s001_100 = {}
        normal_s100_200 = {}
        normal_s200_300 = {}
        normal_s300_500 = {}
        normal_s000_400 = {}

        outlier_s001_100 = {}
        outlier_s100_200 = {}
        outlier_s200_300 = {}
        outlier_s300_500 = {}
        outlier_s000_400 = {}

        normal_avg001_100 = {}
        normal_avg100_200 = {}
        normal_avg200_300 = {}
        normal_avg300_500 = {}
        normal_avg000_400 = {}

        outlier_avg001_100 = {}
        outlier_avg100_200 = {}
        outlier_avg200_300 = {}
        outlier_avg300_500 = {}
        outlier_avg000_400 = {}

        for col in features:
            # fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 20))
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

            for fold in os.listdir(self.data_join_path):
                fold_path = os.path.join(self.data_join_path, fold)
                for ifold in os.listdir(fold_path):
                    ifold_path = os.path.join(fold_path, ifold)
                    key = ':'.join([fold, ifold])
                    for file in os.listdir(ifold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        df_asm = pd.read_excel(os.path.join(ifold_path, file), sheet_name='cyl_data')
                        df_asm = df_asm.set_index(self.drop_cols).rolling(window=10, min_periods=3).mean()
                        df_cols = df_asm[[col]].dropna(how='any', axis=0)
                        if col.startswith('Q'):
                            df_col_scal = df_cols
                            df_col_scal = df_col_scal[df_col_scal.abs() < 1.5]
                        elif (col.endswith('RV') or col.endswith('SV')):
                            df_col_scal = df_cols
                            df_col_scal = df_col_scal[df_col_scal.abs() < 0.05]
                        else:
                            df_col_scal = df_cols / df_cols.values[0]
                        df_col_scal_diff = df_col_scal.rolling(window=20, min_periods=5).mean().diff()

                        if fold.endswith('outlier'):
                            # diff features
                            outlier_s001_100[key] = df_col_scal.values[0] - df_col_scal.values[100]
                            outlier_s100_200[key] = df_col_scal.values[100] - df_col_scal.values[200]
                            outlier_s200_300[key] = df_col_scal.values[200] - df_col_scal.values[300]
                            outlier_s300_500[key] = df_col_scal.values[300] - df_col_scal.values[400]
                            outlier_s000_400[key] = df_col_scal.values[0] - df_col_scal.values[400]
                            # avg features
                            outlier_avg001_100[key] = df_col_scal.values[0:100].mean()
                            outlier_avg100_200[key] = df_col_scal.values[100:200].mean()
                            outlier_avg200_300[key] = df_col_scal.values[200:300].mean()
                            outlier_avg300_500[key] = df_col_scal.values[300:400].mean()
                            outlier_avg000_400[key] = df_col_scal.values[0:400].mean()
                            # ax1.plot(delta_dqdv_data, c='k', label=fold)
                            print()
                        else:
                            # diff features
                            normal_s001_100[key] = df_col_scal.values[0] - df_col_scal.values[100]
                            normal_s100_200[key] = df_col_scal.values[100] - df_col_scal.values[200]
                            normal_s200_300[key] = df_col_scal.values[200] - df_col_scal.values[300]
                            normal_s300_500[key] = df_col_scal.values[300] - df_col_scal.values[400]
                            normal_s000_400[key] = df_col_scal.values[0] - df_col_scal.values[400]
                            # avg features
                            normal_avg001_100[key] = df_col_scal.values[0:100].mean()
                            normal_avg100_200[key] = df_col_scal.values[100:200].mean()
                            normal_avg200_300[key] = df_col_scal.values[200:300].mean()
                            normal_avg300_500[key] = df_col_scal.values[300:400].mean()
                            normal_avg000_400[key] = df_col_scal.values[0:400].mean()

                            # ax1.plot(delta_dqdv_data, c='r', label=fold)

            df_n1 = pd.DataFrame.from_dict(normal_s001_100, orient='index')
            df_n2 = pd.DataFrame.from_dict(normal_s100_200, orient='index')
            df_n3 = pd.DataFrame.from_dict(normal_s200_300, orient='index')
            df_n4 = pd.DataFrame.from_dict(normal_s300_500, orient='index')
            df_nall = pd.DataFrame.from_dict(normal_s000_400, orient='index')

            df_o1 = pd.DataFrame.from_dict(outlier_s001_100, orient='index')
            df_o2 = pd.DataFrame.from_dict(outlier_s100_200, orient='index')
            df_o3 = pd.DataFrame.from_dict(outlier_s200_300, orient='index')
            df_o4 = pd.DataFrame.from_dict(outlier_s300_500, orient='index')
            df_oall = pd.DataFrame.from_dict(outlier_s000_400, orient='index')

            df_n1_avg = pd.DataFrame.from_dict(normal_avg001_100, orient='index')
            df_n2_avg = pd.DataFrame.from_dict(normal_avg100_200, orient='index')
            df_n3_avg = pd.DataFrame.from_dict(normal_avg200_300, orient='index')
            df_n4_avg = pd.DataFrame.from_dict(normal_avg300_500, orient='index')
            df_nall_avg = pd.DataFrame.from_dict(normal_avg000_400, orient='index')

            df_o1_avg = pd.DataFrame.from_dict(outlier_avg001_100, orient='index')
            df_o2_avg = pd.DataFrame.from_dict(outlier_avg100_200, orient='index')
            df_o3_avg = pd.DataFrame.from_dict(outlier_avg200_300, orient='index')
            df_o4_avg = pd.DataFrame.from_dict(outlier_avg300_500, orient='index')
            df_oall_avg = pd.DataFrame.from_dict(outlier_avg000_400, orient='index')
            # @@@@@@@  diff @@@@@@@@@

            ax1.plot(df_n1, c='k', label='0-100:OK', marker='*')
            ax1.plot(df_o1, c='k', label='0-100:NG', marker='o', linewidth=5)

            ax1.plot(df_n2, c='m', label='100-200:OK', marker='*')
            ax1.plot(df_o2, c='m', label='100-200:NG', marker='o', linewidth=5)

            ax1.plot(df_n3, c='y', label='200-300:OK', marker='*')
            ax1.plot(df_o3, c='y', label='200-300:NG', marker='o', linewidth=5)

            ax1.plot(df_n4, c='g', label='300-500:OK', marker='*')
            ax1.plot(df_o4, c='g', label='300-500:NG', marker='o', linewidth=5)

            ax1.plot(df_nall, c='c', label='000-400:OK', marker='*')
            ax1.plot(df_oall, c='c', label='000-400:NG', marker='o', linewidth=5)

            # @@@@@@@  avg @@@@@@@@@
            ax2.plot(df_n1_avg, c='k', label='0-100:OK', marker='*')
            ax2.plot(df_o1_avg, c='k', label='0-100:NG', marker='o', linewidth=5)

            ax2.plot(df_n2_avg, c='m', label='100-200:OK', marker='*')
            ax2.plot(df_o2_avg, c='m', label='100-200:NG', marker='o', linewidth=5)

            ax2.plot(df_n3_avg, c='y', label='200-300:OK', marker='*')
            ax2.plot(df_o3_avg, c='y', label='200-300:NG', marker='o', linewidth=5)

            ax2.plot(df_n4_avg, c='g', label='300-500:OK', marker='*')
            ax2.plot(df_o4_avg, c='g', label='300-500:NG', marker='o', linewidth=5)

            ax2.plot(df_nall_avg, c='c', label='000-400:OK', marker='*')
            ax2.plot(df_oall_avg, c='c', label='000-400:NG', marker='o', linewidth=5)

            # @@@@@@@  note @@@@@@@@@
            ax1.set_title(col + '_OK vs outlier_dqdv_diff', fontsize=24)
            ax2.set_title(col + '_OK vs outlier_dqdv_avg', fontsize=24)
            xtick_label = df_n1.index.tolist() + df_o1.index.tolist()
            xtick_label = [i.split(':')[0] for i in xtick_label]
            ax1.set_xticks(range(0, len(xtick_label)))
            ax1.set_xticklabels(xtick_label, fontsize=16, rotation=-90)
            ax1.legend(loc='best')
            ax1.grid()

            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.legend(loc='best')
            ax2.grid()

            path_dqdv_diff = os.path.join(self.path_png_plt, 'dqdv_diff')
            if not os.path.exists(path_dqdv_diff):
                os.makedirs(path_dqdv_diff)

            plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, wspace=0, hspace=0.35)
            plt.savefig(os.path.join(path_dqdv_diff, col + '_diff-avg.png'), dpi=self.dpi)
            plt.close()
        print()

    def outlier_ts(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        data_join_path = r'E:\code_git\郑州品质_跳水_数据\0.5C\model_data\data_join'
        for sys_fold in os.listdir(data_join_path):
            sys_fold_path = os.path.join(data_join_path, sys_fold)
            for sub_fold in os.listdir(sys_fold_path):
                sub_fold_path = os.path.join(sys_fold_path, sub_fold)
                for file in os.listdir(sub_fold_path):
                    # if not file.endswith('summary.xlsx'):
                    if not file.endswith('220505_038_5-OUTLIER_summary.xlsx'):
                        continue
                    fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(30, 20))

                    file_path = os.path.join(sub_fold_path, file)
                    features = ['CYCLE_NUM', 'END_CAPACITY',
                                'CHG_ENDCAPACITY', 'CHG_ENDENERGY', 'CHG_INITVOL', 'CHG_AVGVOL',
                                'STAT_ENDVOL', 'DELTA_STATVOL', 'MEAN_CHGVOL', 'MEAN_DISCHGVOL']

                    df_ts = pd.read_excel(file_path, sheet_name='cyl_data', index_col='CYCLE_INT').loc[:, features]
                    # df_ts['CHG_AVGVOL_DF1'] = df_ts['CHG_AVGVOL'].diff()
                    # df_ts['CHG_AVGVOL_DF2'] = df_ts['CHG_AVGVOL_DF1'].diff()
                    # df_ts['DELTA_STATVOL_DF1'] = df_ts['DELTA_STATVOL'].diff()
                    # df_ts['DELTA_STATVOL_DF2'] = df_ts['DELTA_STATVOL_DF1'].diff()

                    df_ts.dropna(how='any', axis=0, inplace=True)
                    clf = IsolationForest(n_estimators=20,
                                          max_samples='auto',
                                          contamination=50 / 2000,
                                          max_features=1.0)
                    # clf = LocalOutlierFactor(n_neighbors=10, novelty=False)
                    df_ts['label'] = clf.fit_predict(df_ts)

                    outl_ts = df_ts[df_ts['label'] == -1]

                    path = r'E:\code_git\郑州品质_跳水_数据\0.5C\model_data\data_join\G25_CG_22D25TA-C2(17)-outlier\G25_CG_22D25TA-1#-1-2#220505_038_5-OUTLIER'
                    df_ts.to_excel(os.path.join(path, 'df_outlier.xlsx'))

                    for out_col in features:
                        df_ts.loc[df_ts['label'] == -1, out_col] = np.nan
                        # df_ts[out_col] = df_ts[out_col].fillna(df_ts[out_col].rolling(5, min_periods=1).mean())
                    df_ts = df_ts.reset_index()
                    df_ts.dropna(how='any', axis=0, inplace=True)
                    ax1.plot(df_ts['CYCLE_INT'], df_ts['CHG_AVGVOL'], c='k', label='G25', linewidth=1.0)
                    ax2.plot(df_ts['CYCLE_INT'], df_ts['CHG_INITVOL'], c='k', label='G25', linewidth=1.0)
                    ax3.plot(df_ts['CYCLE_INT'], df_ts['MEAN_CHGVOL'], c='k', label='G25', linewidth=1.0)

                    ax4.plot(df_ts['CYCLE_INT'], df_ts['DELTA_STATVOL'], c='k', label='G25', linewidth=1.0)
                    # ax5.plot(df_ts['CYCLE_INT'], df_ts['CHG_AVGVOL_DF2'], c='k', label='G25', linewidth=1.0)

                    # ax3.plot(df_ts['CYCLE_INT'], df_ts['CHG_INITVOL_DF2'], c='k', label='G25', linewidth=1.0)
                    # ax4.plot(df_ts['CYCLE_INT'], df_ts['CHG_AVGVOL_DF1'], c='k', label='G25', linewidth=1.0)
                    # ax5.plot(df_ts['CYCLE_INT'], df_ts['CHG_AVGVOL_DF2'], c='k', label='G25', linewidth=1.0)

                    ax6.plot(df_ts['CYCLE_INT'], df_ts['DELTA_STATVOL'], c='k', label='G25', linewidth=1.0)
                    ax1.scatter(outl_ts['CYCLE_INT'], outl_ts['CHG_AVGVOL'], c='r', marker='o', s=25)

                    png_fold = os.path.join(self.path_png_plt, 'outlier_fold')
                    if not os.path.exists(png_fold):
                        os.makedirs(png_fold)
                    plt.savefig(os.path.join(png_fold, file.split('.')[0] + '.png'))
                    plt.close()

        # plt.show()
        print()
        pass

    def plt_cmp_allbysysid(self):
        # features_pt = [
        #     'END_CAPACITY', 'DISCHG_ENDENERGY', 'DISCHG_INITVOL', 'DISCHG_AVGVOL',
        #     'CHG_ENDCAPACITY', 'CHG_ENDENERGY', 'CHG_INITVOL', 'CHG_AVGVOL',
        #     'STAT_ENDVOL', 'DELTA_STATVOL',
        #     'MEAN_CHGVOL', 'MEAN_DISCHGVOL', 'RV_VOLTAGE', 'SV_VOLTAGE', 'delta_RV', 'delta_SV',
        #     'Q_EPSILON', 'CC_EPSILON', 'CYCLE_INT', 'fst_peak', 'sec_peak',
        #     'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23',
        #     'MEAN_CHGVOL',
        #
        # ]
        with open(os.path.join(self.pkl_path_mdl, 'lgbm' + '_selected_features.pkl'), 'rb') as sf:
            features = pickle.load(sf)

        for col in features:
            # if not (col.startswith('Q') or col.endswith('RV') or col.endswith('SV')):
            #     continue
            # if not ('pv' in col):
            #     continue
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            k = 1
            for fold in os.listdir(self.data_join_path):
                fold_path = os.path.join(self.data_join_path, fold)
                for ifold in os.listdir(fold_path):
                    ifold_path = os.path.join(fold_path, ifold)
                    for file in os.listdir(ifold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        try:
                            df_asm = pd.read_excel(os.path.join(ifold_path, file), sheet_name='cyl_data')
                            df_asm = df_asm.set_index(self.drop_cols).rolling(window=10, min_periods=3).mean()
                            df_cols = df_asm[[col]].dropna(how='any', axis=0)
                            if col.startswith('Q'):
                                df_col_scal = df_cols
                                df_col_scal = df_col_scal[df_col_scal.abs() < 1.5]
                            elif (col.endswith('RV') or col.endswith('SV')):
                                df_col_scal = df_cols
                                df_col_scal = df_col_scal[df_col_scal.abs() < 0.05]
                            else:
                                df_col_scal = df_cols / df_cols.values[0]
                        except Exception as err:
                            print('fold:{},scaled errors:{}'.format(fold, err))
                            continue

                        # df_col_scal_diff = df_col_scal.rolling(window=20, min_periods=5).mean().diff()
                        df_col_scal_diff = df_cols

                        try:
                            # if (fold.startswith('SC14') and fold.endswith('OK')):
                            #     ax1.plot(df_asm[col].values, c='navy', label=fold)
                            # elif (fold.startswith('SC18') and fold.endswith('OK')):
                            #     ax1.plot(df_asm[col].values, c='k', label=fold)
                            # elif (fold.startswith('SC16') and fold.endswith('OK')):
                            #     ax1.plot(df_asm[col].values[:-2], c='darkblue', label=fold)
                            # elif (fold.startswith('SC5') and fold.endswith('OK')):
                            #     ax1.plot(df_asm[col].values[:-2], c='navy', label=fold)
                            # elif fold.startswith('SC17'):
                            #     ax1.plot(df_asm[col].values[:-2], c='m', label=fold)
                            if fold.endswith('outlier'):
                                if fold.startswith('G25'):
                                    ax1.plot(df_col_scal.values[:-2], c='k', label=fold, linewidth=4)
                                    ax2.plot(df_col_scal_diff.values[:-2], c='k', label=fold, linewidth=4)

                                elif fold.startswith('G24'):
                                    ax1.plot(df_col_scal.values[:-2], c='dimgrey', label=fold, linewidth=4)
                                    ax2.plot(df_col_scal_diff.values[:-2], c='dimgrey', label=fold, linewidth=4)

                                elif fold.startswith('I26'):
                                    ax1.plot(df_col_scal.values[:-2], c='forestgreen', label=fold, linewidth=4)
                                    ax2.plot(df_col_scal_diff.values[:-2], c='forestgreen', label=fold, linewidth=4)

                                else:
                                    ax1.plot(df_col_scal.values[:-2], c='blue', label=fold, linewidth=4)
                                    ax2.plot(df_col_scal_diff.values[:-2], c='blue', label=fold, linewidth=4)
                            else:
                                if fold.startswith('SC9'):
                                    ax1.plot(df_col_scal.values[:-2], c='m', label=fold)
                                    ax2.plot(df_col_scal_diff.values[:-2], c='m', label=fold)

                                else:
                                    ax1.plot(df_col_scal.values, c=self.color_rbow[k], label=fold)
                                    ax2.plot(df_col_scal_diff.values, c=self.color_rbow[k], label=fold)

                        except Exception as err:
                            print('features:{} not in.err:{} '.format(col, err))
                k = k + 20
            path_features_sv = os.path.join(self.path_png_plt, 'features_cmp_fold')
            if not os.path.exists(path_features_sv):
                os.makedirs(path_features_sv)
            if '/' in col:
                col = col.replace('/', '-')
            ax1.set_title('{} '.format(col), fontsize=16)
            ax1.legend(fontsize=6)
            ax2.set_title('{} '.format(col), fontsize=16)
            ax2.legend(fontsize=6)
            plt.savefig(os.path.join(path_features_sv, col + '_.png'), dpi=self.dpi)
            plt.close()

    def plt_cmp_QVbysysid(self):

        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        k = 1
        for fold in os.listdir(self.data_join_path):
            fold_path = os.path.join(self.data_join_path, fold)
            for ifold in os.listdir(fold_path):
                ifold_path = os.path.join(fold_path, ifold)
                for file in os.listdir(ifold_path):
                    if not file.endswith('summary.xlsx'):
                        continue
                    df_asm = pd.read_excel(os.path.join(ifold_path, file), sheet_name='cyl_data')
                    df_asm = df_asm.set_index(self.drop_cols).iloc[self.dqdv_initcycle:, :]

                    try:
                        if (fold.startswith('SC14') and fold.endswith('OK')):
                            ax1.plot(df_asm['CHG_ENDCAPACITY'].values, df_asm['CHG_AVGVOL'].values, c='r', label=fold)
                            ax2.plot(df_asm['DISCHG_AVGVOL'].values, df_asm['END_CAPACITY'].values, c='r', label=fold)

                        elif (fold.startswith('SC18') and fold.endswith('OK')):
                            ax1.plot(df_asm['CHG_ENDCAPACITY'].values, df_asm['CHG_AVGVOL'].values, c='k', label=fold)
                            ax2.plot(df_asm['DISCHG_AVGVOL'].values, df_asm['END_CAPACITY'].values, c='k', label=fold)

                        elif (fold.startswith('SC16') and fold.endswith('OK')):
                            ax1.plot(df_asm['CHG_ENDCAPACITY'].values, df_asm['CHG_AVGVOL'].values, c='teal',
                                     label=fold)
                            ax2.plot(df_asm['DISCHG_AVGVOL'].values, df_asm['END_CAPACITY'].values, c='teal',
                                     label=fold)

                        elif (fold.startswith('SC5') and fold.endswith('OK')):
                            ax1.plot(df_asm['CHG_ENDCAPACITY'].values, df_asm['CHG_AVGVOL'].values, c='navy',
                                     label=fold)
                            ax2.plot(df_asm['DISCHG_AVGVOL'].values, df_asm['END_CAPACITY'].values, c='navy',
                                     label=fold)

                        elif fold.startswith('SC17'):
                            ax1.plot(df_asm['CHG_ENDCAPACITY'].values, df_asm['CHG_AVGVOL'].values, c='m', label=fold)
                            ax2.plot(df_asm['DISCHG_AVGVOL'].values, df_asm['END_CAPACITY'].values, c='m', label=fold)

                        else:
                            ax1.plot(df_asm['CHG_ENDCAPACITY'].values, df_asm['CHG_AVGVOL'].values,
                                     c=self.color_rbow[k], label=fold)
                            ax2.plot(df_asm['DISCHG_AVGVOL'].values, df_asm['END_CAPACITY'].values,
                                     c=self.color_rbow[k], label=fold)

                    except Exception as err:
                        print('features:{} not in.err:{} '.format(' ', err))
            k = k + 10
        path_features_sv = os.path.join(self.path_png_plt, 'features_cmp_fold')
        if not os.path.exists(path_features_sv):
            os.makedirs(path_features_sv)

        ax1.set_title('{} '.format('Q-V'), fontsize=16)
        ax1.legend()
        plt.savefig(os.path.join(path_features_sv, 'QV_.png'), dpi=self.dpi)
        plt.close()

    def plt_cap_corr(self):
        """
        1.plot cap_corr factors
        max pixel:  pixels too large: be less than 2^16 (65536) in each direction
        :return:
        """
        # sns.pairplot(self.df_src)
        fig, (ax1) = plt.subplots(1, 1, figsize=(40, 30))
        plt.tick_params(labelsize=25)
        plt.axvline(x=0, color='black', linestyle='-')
        self.df_corr_spman.sort_values(by=[self.y_label], ascending=False, inplace=True)
        sns.barplot(ax=ax1, x=self.df_corr_spman[self.y_label], y=self.df_corr_spman.index)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=35)
        # ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=40)
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=35)
        ax1.set_title(self.y_label + ' cap_corr factors', fontdict={'weight': 'normal', 'size': 45})
        plt.savefig(os.path.join(self.path_png_plt, 'capacity_raw_corr.png'), dpi=self.dpi)
        # plt.show()
        plt.close()

    def plt_heatmap_corr(self):
        """
        2.plot heatmap figures
        max pixel:  pixels too large: be less than 2^16 (65536) in each direction
        :return:
        """
        fig, (ax2) = plt.subplots(1, 1, figsize=(25, 25))
        cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
        df_copy = self.df_corr_spman.copy(deep=True)
        sns.heatmap(df_copy, cmap=cmap, annot=True, vmax=1, square=True, ax=ax2, annot_kws={"size": 14})
        ax2.set_title('spearman_corr capacity', fontdict={'weight': 'normal', 'size': 35})
        plt.tick_params(labelsize=25)
        label_x = ax2.get_xticklabels()
        label_y = ax2.get_yticklabels()
        plt.setp(label_x, rotation=90)
        plt.setp(label_y, rotation=0)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.2)
        # plt.show()
        plt.savefig(os.path.join(self.path_png_plt, 'capacity_corr_heatmap.png'), dpi=self.dpi)
        plt.close()

    def plt_feature_imp(self):
        fig, (ax1) = plt.subplots(1, 1, figsize=(30, 20))
        sns.barplot(x=self.df_imp['f_importance'], y=self.df_imp['feature_names'], ax=ax1)
        ax1.set_xticklabels(self.df_imp['f_importance'].sort_values(ascending=True), fontsize=18)
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=18)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=35)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=35)
        ax1.set_title(self.y_label + ' cap_corr factors', fontdict={'weight': 'normal', 'size': 45})
        plt.savefig(os.path.join(self.path_png_plt, 'f_importance.png'), dpi=600)
        # plt.show()
        plt.close()

    def plt_trend1_u1(self):
        fig1 = plt.figure(figsize=(width, height))
        ax1 = fig1.add_subplot(131)
        ax1_right = fig1.add_subplot(132)
        ax1_t = fig1.add_subplot(133)
        k = 1
        for fold_tr_pred in os.listdir(self.data_join_path):
            tr_path = os.path.join(self.data_join_path, fold_tr_pred)
            for sysfold in os.listdir(tr_path):
                sysfold_path = os.path.join(tr_path, sysfold)
                for file_fold in os.listdir(sysfold_path):
                    file_fold_path = os.path.join(sysfold_path, file_fold)
                    for file in os.listdir(file_fold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        file_path = os.path.join(file_fold_path, file)
                        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data').set_index(self.drop_cols)
                        df_cyl = df_cyl.rolling(window=10, min_periods=3).mean().iloc[self.dqdv_initcycle:, :]
                        ax1.plot(df_cyl['END_CAPACITY'].abs().values, c=self.color_rbow[k],
                                 label=sysfold + ':' + file_fold)
                        ax1_right.plot(df_cyl['CHG_ENDCAPACITY'].abs().values, c=self.color_vtd[k],
                                       label=sysfold + ':' + file_fold)
                        # ax1_t.plot(df_cyl['DELTA_STATVOL'][df_cyl['DELTA_STATVOL'] > 0].values * 1000,
                        #            c=self.color_sq[k], label=sysfold + ':' + file_fold)
                        ax1_t.plot(df_cyl['DELTA_STATVOL'].values * 1000, c=self.color_sq[k],
                                   label=sysfold + ':' + file_fold)

                        k = k + 10
        xtick = list(range(0, df_cyl.shape[0], self.xtick_step))
        df_cyl.reset_index(inplace=True)
        xtick_label = df_cyl.loc[:, 'CYCLE_NUM'].values[xtick]
        ax1.set_title('dischg_capacity', fontsize=16)
        ax1_t.set_title('delta_u', fontsize=16)
        ax1.set_ylabel('CAPACITY(AH)', fontsize=12)

        ax1_right.set_title('chg_capacity', fontsize=16)
        ax1_right.set_ylabel('CHG_CAPACITY(AH)', fontsize=12)
        ax1_t.set_ylabel('delta_u(mv)', fontsize=12)
        # ax1.set_ylim((97, 108))
        # ax1_t.set_ylim((42, 55))
        ax1.set_xticks(xtick)
        ax1.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax1_right.set_xticks(xtick)
        ax1_right.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax1_t.set_xticks(xtick)
        ax1_t.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax1.legend(fontsize=10, loc='best')
        ax1_t.legend(fontsize=10, loc='best')
        plt.savefig(os.path.join(self.path_png_plt, 'trend1_q-deltau.png'), dpi=self.dpi)

    def plt_trend1_u2(self):
        fig2 = plt.figure(figsize=(width, height))
        ax2 = fig2.add_subplot(121)
        ax2_t = fig2.add_subplot(122)
        k = 1
        for fold_tr_pred in os.listdir(self.data_join_path):
            tr_path = os.path.join(self.data_join_path, fold_tr_pred)
            for sysfold in os.listdir(tr_path):
                sysfold_path = os.path.join(tr_path, sysfold)
                for file_fold in os.listdir(sysfold_path):
                    file_fold_path = os.path.join(sysfold_path, file_fold)
                    for file in os.listdir(file_fold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        file_path = os.path.join(file_fold_path, file)
                        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data').set_index(self.drop_cols)
                        df_cyl = df_cyl.rolling(window=10, min_periods=3).mean().iloc[self.dqdv_initcycle:, :]
                        ax2.plot(df_cyl['DISCHG_INITVOL'].abs().values, c=self.color_rbow[k],
                                 label=sysfold + ':' + file_fold)
                        ax2_t.plot(df_cyl['CHG_INITVOL'].values, c=self.color_sq[k], label=sysfold + ':' + file_fold)
                        k = k + 10
        xtick = list(range(0, df_cyl.shape[0], self.xtick_step))
        df_cyl.reset_index(inplace=True)
        xtick_label = df_cyl.loc[:, 'CYCLE_NUM'].values[xtick]
        ax2.set_title('DISCHG_INITVOL', fontsize=16)
        ax2_t.set_title('CHG_INITVOL', fontsize=16)
        ax2.set_ylabel('DISCHG_INITVOL(v)', fontsize=12)
        ax2_t.set_ylabel('CHG_INITVOL(v)', fontsize=12)
        ax2.set_xticks(xtick)
        ax2.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax2_t.set_xticks(xtick)
        ax2_t.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax2.legend(fontsize=10, loc='best')
        ax2_t.legend(fontsize=10, loc='best')
        plt.savefig(os.path.join(self.path_png_plt, 'trend1_vol_init.png'), dpi=self.dpi)

    def plt_trend1_u3(self):
        fig3 = plt.figure(figsize=(width, height))
        ax3 = fig3.add_subplot(121)
        ax3_t = fig3.add_subplot(122)
        k = 1
        for fold_tr_pred in os.listdir(self.data_join_path):
            tr_path = os.path.join(self.data_join_path, fold_tr_pred)
            for sysfold in os.listdir(tr_path):
                sysfold_path = os.path.join(tr_path, sysfold)
                for file_fold in os.listdir(sysfold_path):
                    file_fold_path = os.path.join(sysfold_path, file_fold)
                    for file in os.listdir(file_fold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        file_path = os.path.join(file_fold_path, file)
                        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data').set_index(self.drop_cols)
                        df_cyl = df_cyl.rolling(window=10, min_periods=3).mean().iloc[self.dqdv_initcycle:, :]
                        ax3.plot(df_cyl['DISCHG_AVGVOL'].abs().values, c=self.color_rbow[k],
                                 label=sysfold + ':' + file_fold)
                        ax3_t.plot(df_cyl['CHG_AVGVOL'].values, c=self.color_sq[k], label=sysfold + ':' + file_fold)
                        k = k + 10
        xtick = list(range(0, df_cyl.shape[0], self.xtick_step))
        df_cyl.reset_index(inplace=True)
        xtick_label = df_cyl.loc[:, 'CYCLE_NUM'].values[xtick]

        ax3.set_title('DISCHG_AVGVOL', fontsize=16)
        ax3_t.set_title('CHG_AVGVOL', fontsize=16)
        ax3.set_ylabel('DISCHG_AVGVOL(v)', fontsize=12)
        ax3_t.set_ylabel('CHG_AVGVOL(v)', fontsize=12)
        ax3.set_xticks(xtick)
        ax3.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax3_t.set_xticks(xtick)
        ax3_t.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax3.legend(fontsize=10, loc='best')
        ax3_t.legend(fontsize=10, loc='best')
        plt.savefig(os.path.join(self.path_png_plt, 'trend1_vol_avg.png'), dpi=self.dpi)

    def plt_trend2_dqdv_ux(self):
        fig1, (ax1_1, ax1_2, ax1_3) = plt.subplots(1, 3, figsize=(15, 8))
        k = 1
        for fold_tr_pred in os.listdir(self.data_join_path):
            tr_path = os.path.join(self.data_join_path, fold_tr_pred)
            for sysfold in os.listdir(tr_path):
                sysfold_path = os.path.join(tr_path, sysfold)
                for file_fold in os.listdir(sysfold_path):
                    file_fold_path = os.path.join(sysfold_path, file_fold)
                    for file in os.listdir(file_fold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        file_path = os.path.join(file_fold_path, file)
                        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data').set_index(self.drop_cols)
                        df_cyl = df_cyl.rolling(window=10, min_periods=3).mean().dropna(how='any', axis=0)
                        ax1_1.plot(df_cyl['fst_peak_x'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        ax1_2.plot(df_cyl['sec_peak_x'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        # ax1_3.plot(df_cyl['thd_peak_x'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        k = k + 1

        xtick = list(range(0, df_cyl.shape[0], self.xtick_step))
        xtick_label = df_cyl.reset_index()['CYCLE_NUM'].values[xtick]
        ax1_1.set_ylabel('fst_peak_xwidth', fontsize=16)
        ax1_2.set_ylabel('sec_peak_xwidth', fontsize=16)
        ax1_3.set_ylabel('thd_peak_xwidth', fontsize=16)

        ax1_1.set_title('fst_peak_xwidth', fontsize=20)
        ax1_2.set_title('sec_peak_xwidth', fontsize=20)
        ax1_3.set_title('thd_peak_xwidth', fontsize=20)

        ax1_1.set_xticks(xtick)
        ax1_1.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax1_2.set_xticks(xtick)
        ax1_2.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax1_3.set_xticks(xtick)
        ax1_3.set_xticklabels(xtick_label, fontsize=12, rotation=45)

        ax1_1.legend(fontsize=10, loc='best')
        ax1_2.legend(fontsize=10, loc='best')
        ax1_3.legend(fontsize=10, loc='best')

        plt.savefig(os.path.join(self.path_png_plt, 'trend2_DQDV_width_u.png'), dpi=self.dpi)

    def plt_trend2_dqdv_yheight(self):
        fig2, (ax2_1, ax2_2, ax2_3) = plt.subplots(1, 3, figsize=(15, 8))
        k = 1
        for fold_tr_pred in os.listdir(self.data_join_path):
            tr_path = os.path.join(self.data_join_path, fold_tr_pred)
            for sysfold in os.listdir(tr_path):
                sysfold_path = os.path.join(tr_path, sysfold)
                for file_fold in os.listdir(sysfold_path):
                    file_fold_path = os.path.join(sysfold_path, file_fold)
                    for file in os.listdir(file_fold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        file_path = os.path.join(file_fold_path, file)
                        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data').set_index(self.drop_cols)
                        df_cyl = df_cyl.rolling(window=10, min_periods=3).mean().dropna(how='any', axis=0)

                        ax2_1.plot(df_cyl['fst_peak'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        ax2_2.plot(df_cyl['sec_peak'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        # ax2_3.plot(df_cyl['thd_peak'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        k = k + 1

        xtick = list(range(0, df_cyl.shape[0], self.xtick_step))
        xtick_label = df_cyl.reset_index()['CYCLE_NUM'].values[xtick]
        ax2_1.set_ylabel('fst_peak_yhight', fontsize=16)
        ax2_2.set_ylabel('sec_peak_yhight', fontsize=16)
        ax2_3.set_ylabel('thd_peak_yhight', fontsize=16)

        ax2_1.set_title('fst_peak_xhight', fontsize=20)
        ax2_2.set_title('sec_peak_xhight', fontsize=20)
        ax2_3.set_title('thd_peak_xhight', fontsize=20)

        ax2_1.set_xticks(xtick)
        ax2_1.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax2_2.set_xticks(xtick)
        ax2_2.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax2_3.set_xticks(xtick)
        ax2_3.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax2_1.legend(fontsize=10, loc='best')
        ax2_2.legend(fontsize=10, loc='best')
        ax2_3.legend(fontsize=10, loc='best')
        # @@@@@@@@@@@@@@@@@@@@@#
        plt.savefig(os.path.join(self.path_png_plt, 'trend2_DQDV_height.png'), dpi=self.dpi)

    def plt_trend2_dqdv_areas(self):
        fig3, (ax3_1, ax3_2, ax3_3) = plt.subplots(1, 3, figsize=(15, 8))
        k = 1
        for fold_tr_pred in os.listdir(self.data_join_path):
            tr_path = os.path.join(self.data_join_path, fold_tr_pred)
            for sysfold in os.listdir(tr_path):
                sysfold_path = os.path.join(tr_path, sysfold)
                for file_fold in os.listdir(sysfold_path):
                    file_fold_path = os.path.join(sysfold_path, file_fold)
                    for file in os.listdir(file_fold_path):
                        if not file.endswith('summary.xlsx'):
                            continue
                        file_path = os.path.join(file_fold_path, file)
                        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data').set_index(self.drop_cols)
                        df_cyl = df_cyl.rolling(window=10, min_periods=3).mean().dropna(how='any', axis=0)

                        if self.MODEL_ID.startswith('LFP'):
                            ax3_1.plot(df_cyl['area_q1'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                            ax3_2.plot(df_cyl['area_q2'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                            ax3_3.plot(df_cyl['area_q3'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        elif self.MODEL_ID.startswith('NC'):
                            ax3_1.plot(df_cyl['area_q1'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                            ax3_2.plot(df_cyl['area1_q23'].values, c=self.color_vtd[k], label=sysfold + ':' + file_fold)
                        k = k + 1

        xtick = list(range(0, df_cyl.shape[0], self.xtick_step))
        xtick_label = df_cyl.reset_index()['CYCLE_NUM'].values[xtick]

        # @@@@@@@@@@@@@@@@@@@@@ #
        ax3_1.set_ylabel('fst_peak_area1', fontsize=16)
        ax3_2.set_ylabel('sec_peak_area2', fontsize=16)
        ax3_3.set_ylabel('thd_peak_area3', fontsize=16)

        ax3_1.set_title('fst_peak_area1', fontsize=20)
        ax3_2.set_title('sec_peak_area2', fontsize=20)
        ax3_3.set_title('thd_peak_area3', fontsize=20)

        ax3_1.set_xticks(xtick)
        ax3_1.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax3_2.set_xticks(xtick)
        ax3_2.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax3_3.set_xticks(xtick)
        ax3_3.set_xticklabels(xtick_label, fontsize=12, rotation=45)
        ax3_1.legend(fontsize=10, loc='best')
        ax3_2.legend(fontsize=10, loc='best')
        ax3_3.legend(fontsize=10, loc='best')

        plt.savefig(os.path.join(self.path_png_plt, 'trend2_DQDV_areas.png'), dpi=self.dpi)

    def plt_cyl_pred(self):
        # plt_cyl_path = os.path.join(self.pred_path, self.mdl_name)
        # plt_cyl_path = os.path.join(self.pred_path, 'lgbm_PRED_ONLY')
        plt_cyl_path = os.path.join(self.pred_path, 'lgbm_PRED_VALID')
        for scellid in os.listdir(plt_cyl_path):
            file_cyl_path = os.path.join(plt_cyl_path, scellid)
            if not os.path.isdir(file_cyl_path):
                continue

            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            color_rbow = plt.cm.rainbow(np.linspace(0, 50, 205))
            marker = ['*', 'o', 's', 'v', '^', 'p', '^', 'v', 'p', 'd', 'h', '2', '8', '6']
            columns = ['slice_' + str(i) for i in self.x_range]
            df_cyl_pred = pd.DataFrame(columns=columns, index=self.y_range)
            df_true_pred = pd.DataFrame(columns=columns + ['TRUE'], index=self.y_range)
            print('sys_fold:{}'.format(file_cyl_path))
            for cond_file in os.listdir(file_cyl_path):
                if not cond_file.endswith('.xlsx'):
                    continue
                df_res = pd.read_excel(os.path.join(file_cyl_path, cond_file), sheet_name=self.sheet_name)
                (row, col) = int(cond_file.split('-')[1][:-5]), 'slice_' + cond_file.split('-')[0]
                i_df = df_res.sort_values(by='CYCLE_NUM').iloc[-1, :]
                df_cyl_pred.loc[row, col] = round(i_df['MAPE'], 3)
                df_true_pred.loc[row, col] = round(i_df['PRED'], 4)
                df_true_pred.loc[row, 'TRUE'] = round(i_df['TRUE'], 4)

            scellid = scellid.split(' ')[0]
            for m, col in enumerate(columns):
                ax.plot(df_cyl_pred[col], color=color_rbow[m], marker=marker[m], label=scellid + ':' + col)
                ax2.plot(df_true_pred[col], color=color_rbow[m], marker=marker[m], label=scellid + '_pred:' + col)
            ax2.plot(df_true_pred['TRUE'], color='k', marker='d', linewidth=4, label=scellid + '_TRUE')

            ax.set_xlabel('to pred:cycle NUM', fontsize=12)
            ax.set_ylabel('pred MAPE(%)', fontsize=12)
            ax2.set_xlabel('to pred:cycle NUM', fontsize=12)
            # ax2.set_ylabel('capacity fade(%)', fontsize=12)
            ax2.set_ylabel('cycle capacity(mAH)', fontsize=12)

            ymin = np.ceil(df_cyl_pred.min(axis=1).min()).astype(int) - 1
            ymax = 100
            # ax.set_xticks(range(0, df_cyl_pred.shape[0]))
            # ax.set_xticklabels(df_cyl_pred.index, fontsize=10, rotation=45)
            # ax.set_yticks(range(ymin, ymax))
            # ax.set_yticklabels(range(ymin, ymax), fontsize=10, rotation=45)

            ax.set_title(scellid + ':SOH cycle_pred ')
            ax.legend(loc='best', fontsize=12)
            ax2.set_title(scellid + ':capacity fade ratio TRUE vs PRED')
            ax2.legend(loc='best', fontsize=12)

            # ax.set_ylim((ymin, ymax))
            save_path = os.path.join(self.path_png_plt, scellid + '_cyl_mape.png')
            # plt.grid(True, color='b')
            # plt.show()
            plt.savefig(save_path, dpi=self.dpi)
            plt.close()

    def plt_cyl_pred_asm(self):
        """
        1.gen total mape
        2.disp mape separately by all,normal,outlier
        3.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        color_rbow = plt.cm.rainbow(np.linspace(0, 20, 256))
        marker = ['*', 'o', 's', 'v', '^', 'p', '^', 'v', 'p', 'd', 'h', '2', '8', '6']
        columns = ['slice_' + str(i) for i in self.x_range]

        # df_mape_norm = pd.DataFrame(columns=columns + ['NORM'], index=self.y_range)
        all_cond_df = []
        norm_cond_df = []
        outlier_cond_df = []
        plt_cyl_path = os.path.join(self.pred_path, self.mdl_name)
        for scellid in os.listdir(plt_cyl_path):
            file_cyl_path = os.path.join(plt_cyl_path, scellid)
            if not os.path.isdir(file_cyl_path):
                continue
            df_mape_all = pd.DataFrame(columns=columns, index=self.y_range)
            for cond_file in os.listdir(file_cyl_path):
                if not cond_file.endswith('.xlsx'):
                    continue
                df_res = pd.read_excel(os.path.join(file_cyl_path, cond_file), sheet_name=self.sheet_name)
                (row, col) = int(cond_file.split('-')[1][:-5]), 'slice_' + cond_file.split('-')[0]
                # i_df = df_res[df_res['SN'] == scellid].sort_values(by='SORT_DATE').iloc[-1:, :]
                i_df = df_res[df_res['SN'] == scellid.split('_')[0]].sort_values(by='CYCLE_NUM').iloc[-1:, :]
                try:
                    df_mape_all.loc[row, col] = round(i_df['MAPE'].values[0], 3)
                except Exception as err:
                    print('cellid:{},cond:{}'.format(scellid, cond_file))

            # all_cond_df.append(df_mape_all.values.tolist())
            # if 'outlier' in scellid:
            #     norm_cond_df.append(df_mape_all.values.tolist())
            # else:
            #     outlier_cond_df.append(df_mape_all.values.tolist())
            all_cond_df.append(df_mape_all.values)
            if 'outlier' in scellid:
                outlier_cond_df.append(df_mape_all.values)
            else:
                norm_cond_df.append(df_mape_all.values)

        all_cond_df = np.array(all_cond_df).mean(axis=0)
        norm_cond_df = np.array(norm_cond_df).mean(axis=0)
        outlier_cond_df = np.array(outlier_cond_df).mean(axis=0)

        df_mape_all_f = pd.DataFrame(data=all_cond_df, columns=columns, index=self.y_range)
        df_mape_norm_f = pd.DataFrame(data=norm_cond_df, columns=columns, index=self.y_range)
        df_mape_outlier_f = pd.DataFrame(data=outlier_cond_df, columns=columns, index=self.y_range)

        for m, cond_col in enumerate(columns):
            ax1.plot(df_mape_all_f[cond_col], color=color_rbow[m], marker=marker[m],
                     label=':'.join([cond_col, 'mape_', 'all']))
            ax2.plot(df_mape_norm_f[cond_col], color=color_rbow[m], marker=marker[m],
                     label=':'.join([cond_col, 'mape_', 'NORM']))
            ax3.plot(df_mape_outlier_f[cond_col], color=color_rbow[m], marker=marker[m],
                     label=':'.join([cond_col, 'mape_', 'OUTLIER']))

        ax1.set_xlabel('to pred:cycle NUM', fontsize=12)
        ax1.set_ylabel('pred MAPE(%)', fontsize=12)
        ax1.set_title('by all mape')

        ax2.set_xlabel('to pred:cycle NUM', fontsize=12)
        ax2.set_ylabel('capacity MAPE(%)', fontsize=12)
        ax2.set_title('by norm mape')
        # ax2.set_ylabel('cycle capacity(mAH)', fontsize=12)

        ax3.set_xlabel('to pred:cycle NUM', fontsize=12)
        ax3.set_ylabel('capacity MAPE(%)', fontsize=12)
        ax3.set_title('by outlier mape')

        ax1.legend(loc='best', fontsize=12)
        ax2.legend(loc='best', fontsize=12)
        ax3.legend(loc='best', fontsize=12)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        # ax.set_ylim((ymin, ymax))
        save_path = os.path.join(self.path_png_plt, 'asm_mape.png')
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

    def plt_cyl_pred_only(self):
        plt_cyl_path = os.path.join(self.pred_path, self.mdl_name + '_' + self.pred_mode)
        for scellid in os.listdir(plt_cyl_path):
            sys_name, file_fold = scellid.split('--')[0], scellid.split('--')[1]
            src_data_path = os.path.join(self.data_join_path, 'PRED', sys_name, file_fold)
            for src_join_file in os.listdir(src_data_path):
                if not src_join_file.endswith('summary.xlsx'):
                    continue
                df_src_join = pd.read_excel(os.path.join(src_data_path, src_join_file), sheet_name='cyl_data')

            file_cyl_path = os.path.join(plt_cyl_path, scellid)
            if not os.path.isdir(file_cyl_path):
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            columns = ['slice_' + str(i) for i in self.x_range]
            df_cyl_pred = pd.DataFrame(columns=columns, index=self.y_range)
            df_true_pred = pd.DataFrame(columns=columns, index=self.y_range)

            for cond_file in os.listdir(file_cyl_path):
                if not cond_file.endswith('.xlsx'):
                    continue
                df_res = pd.read_excel(os.path.join(file_cyl_path, cond_file), sheet_name=self.sheet_name)
                (row, col) = int(cond_file.split('-')[1][:-5]), 'slice_' + cond_file.split('-')[0]
                i_df = df_res.iloc[-1:, :]
                df_true_pred.loc[row, col] = round(i_df['PRED'].values[0], 4)

            ax.plot(self.y_range, df_true_pred, color='m', marker='*', label=''.join([scellid, ':', 'pred']))
            ax.plot(df_src_join[self.y_label], color='g', marker='o', label=''.join([scellid, ':', 'REAL']))
            ax.set_xlabel('to pred:cycle NUM', fontsize=12)
            ax.set_ylabel('pred MAPE(%)', fontsize=12)

            ax.set_title(scellid + ':SOH cycle_pred ')
            ax.legend(loc='best', fontsize=12)
            save_path = os.path.join(self.path_png_plt, self.pred_mode)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, scellid + '_cyl_mape.png'), dpi=self.dpi)
            plt.close()

    def cut_select(self):
        # src_join_path = r'E:\code_git\郑州品质_跳水_数据\0.5C\model_data\data_join_PRD'
        src_join_path = r'E:\code_git\郑州品质_跳水_数据\0.5C\model_data\data_join'
        columns = ['Unnamed: 0', 'delta_pv1', 'delta_pv2', 'sec_peak', 'sec_peak_x']
        df_count = pd.DataFrame(columns=['diff_0-500_secPeakx', 'mean_0-500_secPeakx',
                                         'diff_0-500_secPeaky', 'mean_0-500_secPeaky',
                                         ])
        for tr_prd in os.listdir(src_join_path):
            if not tr_prd in ('TRAIN', 'PRED', 'PRED_OUTLIER'):
                continue
            if not os.path.isdir(os.path.join(src_join_path, tr_prd)):
                continue
            tr_prd_path = os.path.join(src_join_path, tr_prd)
            for sys_id in os.listdir(tr_prd_path):
                sys_id_path = os.path.join(tr_prd_path, sys_id)
                for cell_id in os.listdir(sys_id_path):
                    cell_id_path = os.path.join(sys_id_path, cell_id)
                    for file in os.listdir(cell_id_path):
                        if not file.endswith('pv.xlsx'):
                            continue
                        file_path = os.path.join(cell_id_path, file)
                        try:
                            df = pd.read_excel(file_path, sheet_name='cyl_data', usecols=columns)
                            df.rename(columns={'Unnamed: 0': 'CYL_NUM'}, inplace=True)
                            df.set_index('CYL_NUM', inplace=True)
                            save_path = os.path.join(src_join_path, sys_id + cell_id + file)
                            df.to_excel(save_path)

                            diff_sec_peak_x = df['sec_peak_x'].values[0] - df['sec_peak_x'].values[500]
                            avg_sec_peak_x = df['sec_peak_x'].values[0:500].mean()

                            diff_sec_peak = df['sec_peak'].values[0] - df['sec_peak'].values[500]
                            avg_sec_peak = df['sec_peak'].values[0:500].mean()

                            df_count.loc[sys_id + cell_id, :] = [diff_sec_peak_x, avg_sec_peak_x,
                                                                 diff_sec_peak, avg_sec_peak]
                            print()
                        except Exception as err:
                            print('fold:{} err:{}'.format(sys_id + cell_id, err))
                            continue
        df_count.to_excel(os.path.join(src_join_path, 'count_diff_peak.xlsx'), sheet_name='cyl_count', index=True)


def plt_all(self):
    # self.plt_dist()
    # self.plt_cap_corr()
    # self.plt_heatmap_corr()
    # self.plt_feature_imp()
    # self.plt_cap_err()
    # self.plt_cap_pie()
    # self.outlier_samples()
    # self.cmp_plt()
    # self.cmp_outlier()
    # self.plt_xgb_tree()
    # plt.show()
    pass


if __name__ == "__main__":
    x_range = np.arange(50, 201, 25)
    y_range = np.arange(1000, 1451, 50)
    pv = PostVisual('lgbm', 'PRED_ONLY', x_range, y_range)
    # pv.plt_dist()
    # pv.plt_heatmap_corr()
    pv.plt_cap_corr()
    pv.plt_feature_imp()
    # pv.plt_cmp_all()
    # pv.plt_dqdv_diff()
    # pv.plt_trend1_u1()
    # pv.plt_trend1_u2()
    # pv.plt_trend1_u3()
    # pv.plt_trend2_dqdv_ux()
    # pv.plt_trend2_dqdv_yheight()
    # pv.plt_trend2_dqdv_areas()
    # pv.plt_cyl_pred()

    pass
