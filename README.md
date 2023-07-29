support 产品号：LFP +NCM
before starting:
1. change your source_data path:
   see utils.params_config.ConfPath.raw_par_path  or source fold '代码-结构详情.pptx'
   eg: raw_par_path = r'E:\code_git\抚州_LFP-3-31'
then run 
2. clp_main_run.py
3. knee_point.py 
 
#####################################
1.背景：
近期版本V1.0：基于充放电曲线，计算循环容量
远期版本：加入工艺制程数据、体系材料数据

2.功能目标：
NCM:
    1.保持率预测：50 cycles预测 1000 cycles 准确率> 95%
    2.跳水点事中预警、事前预测
LFP:
    1.保持率预测：50 cycles预测 1000准确率> 95%
    2.跳水点事中预警、事前预测

3.说明：
#*****************************#
-1.保持率预测：
from xfun.parse_data.parse_func import LFPLoadSmall
from xfun.preclean import gen_features as gf
from xfun.preclean.clean_merge import PreClean
from xfun.preclean.gen_features import ExtractFeatures
from xfun.train_model.corr_features_calcu import FeatureAnsys
from xfun.train_model.models_calcu import ModelTrain
from xfun.pred_model.model_preds import ModelPredict
from xfun.post_plot.plt_mdl_data import PostVisual

    1.class LFPLoadSmall 数据拆分类
      split_cycle             --拆分循环号数据
      gen_pltdata             --生成dqdv数据
    2.class ExtractFeatures dqdv计算类
      get_peak_valley         -- dqdv峰特征提取
    3.class PreClean 训练-预测数据构建类
         lfp_pv_join          -- 特征数据合并
         gen_mdldat           -- 数据集构建
    4.class FeatureAnsys 特征计算类
        features_calcu        --特征计算及抽取排序
    5.class ModelTrain  模型训练类
        fit_mdl               --模型训练
    6.class ModelPredict 模型预测类
        pred_tmp(mdl_name)  --模型训练
    7.class PostVisual  后处理类
        pv.plt_dist()         --数据
        pv.plt_heatmap_corr() --相关性热力图
        pv.plt_cap_corr()     --容量影响因子图
        pv.plt_feature_imp()  --特征重要度
        pv.plt_trend1_u()     --电压特征可视化
        pv.plt_trend2_dqdv()  --dqdv特征可视化
        pv.plt_cyl_pred()     --循环结果可视化
    8. animation save 
        RuntimeError: Requested MovieWriter (ffmpeg) not available
        -1. 下载 ffmpeg ,设置环境变量：D:\ffmpeg_3.4.2\bin\x64
        -2. 安装ffmpeg、ffmpeg-python：pip install ffmpeg ffmpeg-python
        -3. 查看 ffmpeg 是否安装成功
            from matplotlib import animation
            print(animation.writers.list())

#*****************************#
xfun.outlier_kneepoint_detect 

2.跳水点监测：
    kpc = KneeePointCalcu()
    kpc.knee_point_fit()
    kpc.plt_cmp_run()
    kpc.diff_cmp()

3.outlier_kpc


#*****************************#
4.simple run capacity
    xfun.cyl_thin 


