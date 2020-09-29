#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
evaluate multi time-window model taht is fusion model of multi target models
 * select type: "EXP", "VA_V", "VA_A"
 * select sub1, sub2 type: "EXP", "VA_V", "VA_A"
 * evaluate validation per frame
"""
import numpy as np
import pandas as pd
import glob
import sklearn #機械学習のライブラリ
import lightgbm as lgb
from sklearn.metrics import accuracy_score,mean_squared_error,f1_score
from statistics import mean, median,variance,stdev
import math
from scipy import stats
import pickle
import os
import pathlib


# In[2]:


# create base dataset: concat au csv & add label, file count & drop unnecessary columns
#   str_substract_time: ex. '02s' , str_search_key: ex. '(Subject_).*.csv'
#   cut_start: trim data (X sec from start), cut_end: trim data (X sec from end)
def crate_base_data(data_file_names, str_type, str_time):
    # create empty dataframe (base dataframe)
    #data = pd.DataFrame()
    count = 0
    max_count = len(data_file_names)
    
    data_list = [pd.DataFrame()] # <- dummy
    
    for  data_file in data_file_names:
        # read au csv
        if os.path.isfile(data_file) and os.path.getsize(data_file) > 32:
            #print(os.path.getsize(data_file))
            #data_tmp = pd.read_csv(data_file)
            data_tmp = pd.read_hdf(data_file)
        else:
            count = count+1
            continue
        
        if (len(data_tmp)<1):
            count = count+1
            continue
        
        # create column - 'count', 'Label', 'subject' (default: 0)
        data_tmp["count"] = 0
        data_tmp["subject"] = "sample"

        # convert filename to 'subject'
        name_train = os.path.splitext(os.path.basename(data_file))[0].replace(str_time,'')
        #print(name_train)

        #print(data_temp)
        # get and set Label value
        data_tmp["count"]  = count
        data_tmp["subject"] = name_train
        
        # drop unnecessary columns
        # ' frame-avg',' face_id-avg,' timestamp-avg',' confidence-avg,' success-avg','frame-std',' face_id-std',' confidence-std',' success-std'
        data_tmp = data_tmp.drop(['frame-avg',' face_id-avg',' timestamp-avg',' confidence-avg',' success-avg',
                                  'frame-std',' face_id-std',' timestamp-std',' confidence-std',' success-std',
                                  'frame-range', ' face_id-range', ' timestamp-range', ' confidence-range', ' success-range',
                                  'frame-slope', ' face_id-slope', ' timestamp-slope', ' confidence-slope', ' success-slope',
                                  'Unnamed: 0-avg', 'Unnamed: 0-std', 'Unnamed: 0-range', 'Unnamed: 0-slope'
                               ], axis=1)
        if str_type == "EXP":
            data_tmp = data_tmp.drop(['Neutral-std','Neutral-range','Neutral-slope'], axis=1)
        else:
            data_tmp = data_tmp.drop(['arousal-std', 'arousal-range', 'arousal-slope', 
                                     'valence-std', 'valence-range', 'valence-slope'], axis=1)

        # append created data to base dataframe
        #data = data.append(data_tmp)
        data_list.append(data_tmp)

        log = 'count: {0}, name: {1}, data shape: {2}'.format(count, name_train, data_tmp.shape)
        print(log)
        count = count + 1
    # finish
    del data_list[0]
    data = pd.concat([x for x in data_list])
    
    log = '**** finished creating base dataset, data shape: {0}'.format(data.shape)
    print(log)
    
    return data


# In[3]:


def load_model(file_model):
    with open(file_model, mode='rb') as fp:
        model = pickle.load(fp)
    return model


# In[4]:


# split base data to <au>, <gaze and pose>, <eye_landmark, 2d landmark, 3d landmark>
# ** 'count','label','subject' is contained in all splits
def split_data(in_data):
    # au data
    df_au = in_data.loc[:, in_data.columns.str.contains("AU|count|subject|Neutral|valence|arousal") ]
    #df_au = df_au.join(df_lable)
    print("AU data shape: ",df_au.shape)

    # gaze and pose data **** temp pose
    df_pose = in_data.loc[:, in_data.columns.str.contains("pose_|count|subject|Neutral|valence|arousal") ]
    #df_pose = df_pose.join(df_lable)
    print("Gaze & Pose data shape: ",df_pose.shape)
    
    # eye_landmark, 2d landmark, 3d landmark data **** temp gaze
    df_lmk = in_data.loc[:, in_data.columns.str.contains("gaze|count|subject|Neutral|valence|arousal")]
    #df_lmk = df_lmk.join(df_lable)
    print("Landmark data shape: ",df_lmk.shape)
    
    # openpose
    #df_op = in_data.loc[:, ~in_data.columns.str.contains("AU|pose_|gaze")]
    df_op = in_data.loc[:, in_data.columns.str.contains("hand_flag|0x|0y|0c|1x|1y|1c|2x|2y|2c|3x|3y|3c|4x|4y|4c|5x|5y|5c|6x|6y|6c|7x|7y|7c|8x|8y|8c|9x|9y|9c|10x|10y|10c|11x|11y|11c|12x|12y|12c|13x|13y|13c|14x|14y|14c|15x|15y|15c|16x|16y|16c|17x|17y|17c|18x|18y|18c|19x|19y|19c|20x|20y|20c|21x|21y|21c|22x|22y|22c|23x|23y|23c|24x|24y|24c|count|subject|Neutral|valence|arousal")]
    print("Opepose data shape: ",df_op.shape)
    
    # resnet
    df_rn = in_data.loc[:, ~in_data.columns.str.contains("AU|pose_|gaze|hand_flag|0x|0y|0c|1x|1y|1c|2x|2y|2c|3x|3y|3c|4x|4y|4c|5x|5y|5c|6x|6y|6c|7x|7y|7c|8x|8y|8c|9x|9y|9c|10x|10y|10c|11x|11y|11c|12x|12y|12c|13x|13y|13c|14x|14y|14c|15x|15y|15c|16x|16y|16c|17x|17y|17c|18x|18y|18c|19x|19y|19c|20x|20y|20c|21x|21y|21c|22x|22y|22c|23x|23y|23c|24x|24y|24c")]
    print("Resnet data shape: ",df_rn.shape)
    
    print("** end **")
    return df_au,df_pose,df_lmk,df_op, df_rn
    


# In[5]:


# create dataset for single time analysis (Light GBM ...)
def make_dataset_for_gbm(in_data, str_type):
    
    if str_type == "EXP":
        in_data = in_data[in_data["Neutral-avg"]>=0]
        # EXP
        # spplit features , labels
        data_y = in_data.loc[:,["count", "subject", "Neutral-avg"]]
        data_x = in_data.drop(["count", "subject", "Neutral-avg"], axis=1)
    else:
        # VA
        # spplit features , labels
        data_y = in_data.loc[:,["count", "subject", "valence-avg", "arousal-avg"]]
        data_x = in_data.drop(["count", "subject", "valence-avg", "arousal-avg"], axis=1)
    
    dim = len(data_x.columns)
    
    # drop 'count','group' from data_y
    data_y = data_y.drop(["count", "subject"], axis=1) 
    if str_type == "VA_A":
        data_y = data_y.drop(["valence-avg"], axis=1) 
    elif str_type == "VA_V":
        data_y = data_y.drop(["arousal-avg"], axis=1) 
    
    # convert pandas to numpy 
    np_data_x = data_x.values
    np_data_y = data_y.values
    
    # reshape data for tda
    np_data_x = np.reshape(np_data_x, [len(np_data_y),dim])
    np_data_y = np.reshape(np_data_y, [len(np_data_y),1])

    return np_data_x, np_data_y


# In[6]:


# create dataset for single time analysis (Light GBM ...)
def make_dataset_for_gbm_sub(in_data, str_type):
    
    data_x = in_data.drop(["count", "subject"], axis=1)
    if "Neutral-avg" in data_x.columns:
        data_x = data_x.drop(["Neutral-avg"], axis=1)
    if "valence-avg" in data_x.columns:
        data_x = data_x.drop(["valence-avg"], axis=1)
    if "arousal-avg" in data_x.columns:
        data_x = data_x.drop(["arousal-avg"], axis=1)
    
    dim = len(data_x.columns)
    length = len(data_x)
    
    # convert pandas to numpy 
    np_data_x = data_x.values
    
    # reshape data for tda
    np_data_x = np.reshape(np_data_x, [length, dim])
    
    return np_data_x


# In[7]:


# predict
def predict_data_val_time(data_val, models, str_type, window_time, dir_features = None):
    log = "split data to AU ,pose, gaze, openpose"
    print(log)
    val_au, val_pose, val_lmk, val_op, val_rn = split_data(data_val)
    
    if dir_features != None:
        file_f = dir_features + "features_" + str_type +"_au.csv"
        val_au = substruct_features(val_au, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_pose.csv"
        val_pose = substruct_features(val_pose, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_lmk.csv"
        val_lmk = substruct_features(val_lmk, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_op.csv"
        val_op = substruct_features(val_op, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_en.csv"
        val_rn = substruct_features(val_rn, file_f, str_type, window_time)
    
    log = "convert data(pandas) to data(numpy) for LightGBM"
    print(log)
    
    np_val_au_x, np_val_au_y = make_dataset_for_gbm(val_au, str_type)
    np_val_pose_x, np_val_pose_y = make_dataset_for_gbm(val_pose, str_type)
    np_val_lmk_x, np_val_lmk_y = make_dataset_for_gbm(val_lmk, str_type)
    np_val_op_x, np_val_op_y = make_dataset_for_gbm(val_op, str_type)
    np_val_rn_x, np_val_rn_y = make_dataset_for_gbm(val_rn, str_type)

    log = "predict by au, pose, gaze, openpose, GAPP ensemple"
    print(log)
        
    if str_type == "EXP":
        pred_val_au = models[0].predict(np_val_au_x)
        pred_val_pose = models[1].predict(np_val_pose_x)
        pred_val_lmk = models[2].predict(np_val_lmk_x)
        pred_val_op = models[3].predict(np_val_op_x)
        pred_val_rn = models[4].predict(np_val_rn_x)
        np_val_ens_x = np.column_stack((pred_val_au, pred_val_pose, pred_val_lmk,
                                        pred_val_op, pred_val_rn))
        pred_val_ens = models[5].predict(np_val_ens_x)
        np_val_true = np_val_au_y.ravel()
    else:
        pred_val_au = models[0].predict(np_val_au_x).ravel()
        pred_val_pose = models[1].predict(np_val_pose_x).ravel()
        pred_val_lmk = models[2].predict(np_val_lmk_x).ravel()
        pred_val_op = models[3].predict(np_val_op_x).ravel()
        pred_val_rn = models[4].predict(np_val_rn_x).ravel()
        np_val_ens_x = np.column_stack((pred_val_au, pred_val_pose, pred_val_lmk, 
                                        pred_val_op, pred_val_rn))
        pred_val_ens = models[5].predict(np_val_ens_x).ravel()
        np_val_true = np_val_au_y.ravel()
    
    return np_val_true, pred_val_ens


# In[8]:


# predict
def predict_data_val_time_sub(data_val, models, str_type, main_type, window_time, dir_features = None):
    log = "split data to AU ,pose, gaze, openpose"
    print(log)
    val_au, val_pose, val_lmk, val_op, val_rn = split_data(data_val)
    
    if dir_features != None:
        file_f = dir_features + "features_" + str_type +"_au.csv"
        val_au = substruct_features(val_au, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_pose.csv"
        val_pose = substruct_features(val_pose, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_lmk.csv"
        val_lmk = substruct_features(val_lmk, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_op.csv"
        val_op = substruct_features(val_op, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_en.csv"
        val_rn = substruct_features(val_rn, file_f, main_type, window_time)
    
    log = "convert data(pandas) to data(numpy) for LightGBM"
    print(log)
    
    np_val_au_x = make_dataset_for_gbm_sub(val_au, main_type)
    np_val_pose_x = make_dataset_for_gbm_sub(val_pose, main_type)
    np_val_lmk_x = make_dataset_for_gbm_sub(val_lmk, main_type)
    np_val_op_x = make_dataset_for_gbm_sub(val_op, main_type)
    np_val_rn_x = make_dataset_for_gbm_sub(val_rn, main_type)

    log = "predict by au, pose, gaze, openpose, GAPP ensemple"
    print(log)
    print(str_type)
        
    if str_type == "EXP":
        pred_val_au = models[0].predict(np_val_au_x)
        pred_val_pose = models[1].predict(np_val_pose_x)
        pred_val_lmk = models[2].predict(np_val_lmk_x)
        pred_val_op = models[3].predict(np_val_op_x)
        pred_val_rn = models[4].predict(np_val_rn_x)
        np_val_ens_x = np.column_stack((pred_val_au, pred_val_pose, pred_val_lmk,
                                        pred_val_op, pred_val_rn))
        pred_val_ens = models[5].predict(np_val_ens_x)
    else:
        pred_val_au = models[0].predict(np_val_au_x).ravel()
        pred_val_pose = models[1].predict(np_val_pose_x).ravel()
        pred_val_lmk = models[2].predict(np_val_lmk_x).ravel()
        pred_val_op = models[3].predict(np_val_op_x).ravel()
        pred_val_rn = models[4].predict(np_val_rn_x).ravel()
        np_val_ens_x = np.column_stack((pred_val_au, pred_val_pose, pred_val_lmk, 
                                        pred_val_op, pred_val_rn))
        pred_val_ens = models[5].predict(np_val_ens_x).ravel()
    
    return pred_val_ens


# In[9]:


# evaluate predict data
def eval_pred(data_true, data_pred, str_type):
    score_f1 = 0
    score_acc = 0

    # if target is expression, calc F1 score, accuracy
    if str_type == "EXP":
        # convert to 7-columns predict probability to 1-column predict
        pred_tmp = np.argmax(data_pred, axis=1) # 一番大きい予測確率のクラスを予測クラスに
        
        ltrue = list(data_true)
        lpred = list(pred_tmp) 
        
        score_1 = f1_score(ltrue, lpred, average='macro') # weighted, macro, micro
        score_2 = accuracy_score(ltrue, lpred)
    # if target is VA, calc CCC, mse
    else:
        pred_tmp = data_pred
        #pred_tmp = data_pred.round()
        x_mean = pred_tmp.mean()
        y_mean = data_true.mean()
        sx2 = ((pred_tmp-x_mean)*(pred_tmp-x_mean)).mean()
        sy2 = ((data_true-y_mean)*(data_true-y_mean)).mean()
        sxy = ((pred_tmp-x_mean)*(data_true-y_mean)).mean()
        score_1 = (2 * sxy) / (sx2 + sy2 + (x_mean - y_mean) * (x_mean - y_mean))
        #score_acc = (2 * sxy) / (sx2 + sy2 + (x_mean - y_mean) * (x_mean - y_mean))
        score_2 = mean_squared_error(data_true, data_pred)
    return score_1, score_2


# In[10]:


def read_models(dir_model, str_type, window_time):
    ext = "_{0}s.pickle".format(str(window_time).zfill(2))
    
    model_au = load_model(dir_model + "model_au_gbm_" + str_type + ext)
    model_pose = load_model(dir_model + "model_pose_gbm_" + str_type + ext)
    model_lmk = load_model(dir_model + "model_lmk_gbm_" + str_type + ext)
    model_op = load_model(dir_model + "model_op_gbm_" + str_type + ext)
    model_rn = load_model(dir_model + "model_rn_gbm_" + str_type + ext)
    model_ens = load_model(dir_model + "model_ens_gbm_" + str_type + ext)
    
    models = [model_au, model_pose, model_lmk, model_op, model_rn, model_ens]
    
    return models


# In[11]:


def substruct_features(data, fp, str_type, window_time):
    col = str(window_time).zfill(2) + "s"
    data_f = pd.read_csv(fp)
    
    if str_type == "EXP":
        list_f = list(data_f[col].values.ravel())
        list_f = np.append(list_f, ["count", "subject", "Neutral-avg"])
    else:
        list_f = list(data_f[col].values.ravel())
        list_f = np.append(list_f, ["count", "subject", "valence-avg", "arousal-avg"])
    
    data2 = data[list_f]
    
    return data2


# In[1]:


def run_validation_frame(dir_validation, dir_model_01s, dir_model_06s, dir_model_12s, dir_model_03s, dir_model_fusion,
                         dir_out, target, target_sub1, target_sub2, dir_features=None):
    str_type = target
    str_type_sub1 = target_sub1
    str_type_sub2 = target_sub2
    
    log = "Target type: {0}".format(str_type)
    print(log)
    
    # read models
    model_01s = read_models(dir_model_01s, str_type, 1)
    model_06s = read_models(dir_model_06s, str_type, 6)
    model_12s = read_models(dir_model_12s, str_type, 12)
    model_03s = read_models(dir_model_03s, str_type, 3)
    
    model_01s_sub1 = read_models(dir_model_01s, str_type_sub1, 1)
    model_06s_sub1 = read_models(dir_model_06s, str_type_sub1, 6)
    model_12s_sub1 = read_models(dir_model_12s, str_type_sub1, 12)
    model_03s_sub1 = read_models(dir_model_03s, str_type_sub1, 3)
    
    model_01s_sub2 = read_models(dir_model_01s, str_type_sub2, 1)
    model_06s_sub2 = read_models(dir_model_06s, str_type_sub2, 6)
    model_12s_sub2 = read_models(dir_model_12s, str_type_sub2, 12)
    model_03s_sub2 = read_models(dir_model_03s, str_type_sub2, 3)
    
    model_fusion = load_model(dir_model_fusion + "model_fusion_multi_" + str_type + ".pickle")
    model_fusion_sub1 = load_model(dir_model_fusion + "model_fusion_single_" + str_type_sub1 + ".pickle")
    model_fusion_sub2 = load_model(dir_model_fusion + "model_fusion_single_" + str_type_sub2 + ".pickle")
    
    # search files of validation data
    file_val = dir_validation + "*_01s.h5"
    files_val_01s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 01s: {0}".format(len(files_val_01s))
    print(log)

    file_val = dir_validation + "*_06s.h5"
    files_val_06s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 06s: {0}".format(len(files_val_06s))
    print(log)

    file_val = dir_validation + "*_12s.h5"
    files_val_12s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 12s: {0}".format(len(files_val_12s))
    print(log)
    
    file_val = dir_validation + "*_03s.h5"
    files_val_03s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 03s: {0}".format(len(files_val_03s))
    print(log)
    
    # create base dataset
    log = "data loading...."
    print(log)

    str_time = "_01s"
    #data_train = pd.read_hdf(file_train, 'key')
    data_val_01s = crate_base_data(files_val_01s, str_type, str_time)
    log = "data validation 01s shape: {0}".format(data_val_01s.shape)
    print(log)

    str_time = "_06s"
    data_val_06s = crate_base_data(files_val_06s, str_type, str_time)
    log = "data validation 06s shape: {0}".format(data_val_06s.shape)
    print(log)

    str_time = "_12s"
    data_val_12s = crate_base_data(files_val_12s, str_type, str_time)
    log = "data validation 12s shape: {0}".format(data_val_12s.shape)
    print(log)
    
    str_time = "_03s"
    data_val_03s = crate_base_data(files_val_03s, str_type, str_time)
    log = "data validation 03s shape: {0}".format(data_val_03s.shape)
    print(log)

    #data_val = pd.read_hdf(file_val, 'key')

    # create base dataset
    log = "finished data loading"
    print(log)
    
    # adjust data shape (same frame)
    log = "val data shape) 01s: {0}, 06s: {1}, 12s: {2}, 03s: {3}".format(data_val_01s.shape, data_val_06s.shape,
                                                                          data_val_12s.shape, data_val_03s.shape)
    print(log)

    length_columns = len(data_val_01s.columns)
    base_columns = data_val_01s.columns

    data_val_01s.columns = data_val_01s.columns + "_01s"
    data_val_06s.columns = data_val_06s.columns + "_06s"
    data_val_12s.columns = data_val_12s.columns + "_12s"
    data_val_03s.columns = data_val_03s.columns + "_03s"

    data_val = pd.concat([data_val_01s, data_val_06s, data_val_12s, data_val_03s], axis=1)
    if str_type == "EXP":
        data_val = data_val.loc[data_val["Neutral-avg_01s"]>=0]
        data_val = data_val.loc[data_val["Neutral-avg_06s"]>=0]
        data_val = data_val.loc[data_val["Neutral-avg_12s"]>=0]
        data_val = data_val.loc[data_val["Neutral-avg_03s"]>=0]
    data_val = data_val.dropna(how='any')
    val_index = data_val.index

    data_val_01s = data_val.iloc[:,0:length_columns]
    data_val_06s = data_val.iloc[:,length_columns:length_columns*2]
    data_val_12s = data_val.iloc[:,length_columns*2:length_columns*3]
    data_val_03s = data_val.iloc[:,length_columns*3:length_columns*4]

    data_val_01s.columns = base_columns
    data_val_06s.columns = base_columns
    data_val_12s.columns = base_columns
    data_val_03s.columns = base_columns

    log = "val data shape) 01s: {0}, 06s: {1}, 12s: {2}, 03s: {3}".format(data_val_01s.shape, data_val_06s.shape,
                                                                data_val_12s.shape, data_val_03s.shape)
    print(log)

    # main
    # 01s
    window_time = 1
    np_val_true_01s, pred_val_ens_01s = predict_data_val_time(data_val_01s, model_01s, str_type, window_time, dir_features)
    log = "01s pred shape) val: {0}".format(pred_val_ens_01s.shape)
    print(log)

    # 06s
    window_time = 6
    np_val_true_06s, pred_val_ens_06s = predict_data_val_time(data_val_06s, model_06s, str_type, window_time, dir_features)
    log = "06s pred shape) val: {0}".format(pred_val_ens_06s.shape)
    print(log)

    # 12s
    window_time = 12
    np_val_true_12s, pred_val_ens_12s = predict_data_val_time(data_val_12s, model_12s, str_type, window_time, dir_features)
    log = "12s pred shape) val: {0}".format(pred_val_ens_12s.shape)
    print(log)
    
    # 03s
    window_time = 3
    np_val_true_03s, pred_val_ens_03s = predict_data_val_time(data_val_03s, model_03s, str_type, window_time, dir_features)
    log = "03s pred shape) val: {0}".format(pred_val_ens_03s.shape)
    print(log)


    score_1, score_2 = eval_pred(np_val_true_01s, pred_val_ens_01s, str_type)
    if str_type == "EXP":
        score = 0.67*score_1 + 0.33*score_2
    else:
        score = score_1
    log1 = "01s, score: {0}. score1: {1}, score2: {2}".format(score, score_1, score_2)
    score_1, score_2 = eval_pred(np_val_true_01s, pred_val_ens_06s, str_type)
    if str_type == "EXP":
        score = 0.67*score_1 + 0.33*score_2
    else:
        score = score_1
    log2 = "06s, score: {0}. score1: {1}, score2: {2}".format(score, score_1, score_2)
    score_1, score_2 = eval_pred(np_val_true_01s, pred_val_ens_12s, str_type)
    if str_type == "EXP":
        score = 0.67*score_1 + 0.33*score_2
    else:
        score = score_1
    log3 = "12s, score: {0}. score1: {1}, score2: {2}".format(score, score_1, score_2)
    score_1, score_2 = eval_pred(np_val_true_01s, pred_val_ens_03s, str_type)
    if str_type == "EXP":
        score = 0.67*score_1 + 0.33*score_2
    else:
        score = score_1
    log4 = "03s, score: {0}. score1: {1}, score2: {2}".format(score, score_1, score_2)
    

    # sub1 (with fusion single)
    # 01s
    window_time = 1
    #pred_train_01s, pred_val_01s = predict_data(data_train_01s, data_val_01s, model_01s, str_type)
    pred_val_ens_01s_sub1 = predict_data_val_time_sub(data_val_01s, model_01s_sub1, str_type_sub1, str_type, window_time, dir_features)
    log = "01s pred shape) val: {0}".format(pred_val_ens_01s_sub1.shape)
    print(log)

    # 06s
    window_time = 6
    #pred_train_06s, pred_val_06s = predict_data(data_train_06s, data_val_06s, model_06s, str_type)
    pred_val_ens_06s_sub1 = predict_data_val_time_sub(data_val_06s, model_06s_sub1, str_type_sub1, str_type, window_time, dir_features)
    log = "06s pred shape) val: {0}".format(pred_val_ens_06s_sub1.shape)
    print(log)

    # 12s
    window_time = 12
    #pred_train_12s, pred_val_12s = predict_data(data_train_12s, data_val_12s, model_12s, str_type)
    pred_val_ens_12s_sub1 = predict_data_val_time_sub(data_val_12s, model_12s_sub1, str_type_sub1, str_type, window_time, dir_features)
    log = "12s pred shape) val: {0}".format(pred_val_ens_12s_sub1.shape)
    print(log)
    
    # 3s
    window_time = 3
    #pred_train_03s, pred_val_03s = predict_data(data_train_03s, data_val_03s, model_03s, str_type)
    pred_val_ens_03s_sub1 = predict_data_val_time_sub(data_val_03s, model_03s_sub1, str_type_sub1, str_type, window_time, dir_features)
    log = "03s pred shape) val: {0}".format(pred_val_ens_03s_sub1.shape)
    print(log)

    # fusion sub1

    np_val_x1 = pred_val_ens_01s_sub1
    np_val_x2 = pred_val_ens_06s_sub1
    np_val_x3 = pred_val_ens_12s_sub1
    np_val_x4 = pred_val_ens_03s_sub1
    #np_val_y  = np_val_true_01s_sub1


    # stacked predict data: validation
    stack_pred_val = np.column_stack((np_val_x1, np_val_x2, np_val_x3, np_val_x4))

    if str_type_sub1 == "EXP":
        #pred_train_sub1 = model_fusion.predict(stack_pred)
        pred_val_sub1 = model_fusion_sub1.predict(stack_pred_val)
    else:
        #pred_train_sub1 = model_fusion.predict(stack_pred).ravel()
        pred_val_sub1 = model_fusion_sub1.predict(stack_pred_val).ravel()

    # sub2 (with fusion single)
    # 01s
    window_time = 1
    #pred_train_01s, pred_val_01s = predict_data(data_train_01s, data_val_01s, model_01s, str_type)
    pred_val_ens_01s_sub2 = predict_data_val_time_sub(data_val_01s, model_01s_sub2, str_type_sub2, str_type, window_time, dir_features)
    log = "01s pred shape) val: {0}".format(pred_val_ens_01s_sub2.shape)
    print(log)

    # 06s
    window_time = 6
    #pred_train_06s, pred_val_06s = predict_data(data_train_06s, data_val_06s, model_06s, str_type)
    pred_val_ens_06s_sub2 = predict_data_val_time_sub(data_val_06s, model_06s_sub2, str_type_sub2, str_type, window_time, dir_features)
    log = "06s pred shape) val: {0}".format(pred_val_ens_06s_sub2.shape)
    print(log)

    # 12s
    window_time = 12
    #pred_train_12s, pred_val_12s = predict_data(data_train_12s, data_val_12s, model_12s, str_type)
    pred_val_ens_12s_sub2 = predict_data_val_time_sub(data_val_12s, model_12s_sub2, str_type_sub2, str_type, window_time, dir_features)
    log = "12s pred shape) val: {0}".format(pred_val_ens_12s_sub2.shape)
    print(log)
    
    # 3s
    window_time = 3
    #pred_train_03s, pred_val_03s = predict_data(data_train_03s, data_val_03s, model_03s, str_type)
    pred_val_ens_03s_sub2 = predict_data_val_time_sub(data_val_03s, model_03s_sub2, str_type_sub2, str_type, window_time, dir_features)
    log = "03s pred shape) val: {0}".format(pred_val_ens_03s_sub2.shape)
    print(log)

    # fusion sub1

    np_val_x1 = pred_val_ens_01s_sub2
    np_val_x2 = pred_val_ens_06s_sub2
    np_val_x3 = pred_val_ens_12s_sub2
    np_val_x4 = pred_val_ens_03s_sub2
    #np_val_y  = np_val_true_01s_sub2

    # stacked predict data: validation
    stack_pred_val = np.column_stack((np_val_x1, np_val_x2, np_val_x3, np_val_x4))

    if str_type_sub2 == "EXP":
        #pred_train_sub2 = model_fusion.predict(stack_pred)
        pred_val_sub2 = model_fusion_sub2.predict(stack_pred_val)
    else:
        #pred_train_sub2 = model_fusion.predict(stack_pred).ravel()
        pred_val_sub2 = model_fusion_sub2.predict(stack_pred_val).ravel()

    # single ensemble ************** if needed
    # stacked predict data: validation
    stack_pred_val = np.column_stack((pred_val_ens_01s, pred_val_ens_06s, pred_val_ens_12s, pred_val_ens_03s))

    model_fusion_tmp = load_model(dir_model_fusion + "model_fusion_single_" + str_type + ".pickle")

    if str_type == "EXP":
        #pred_train_sub1 = model_fusion.predict(stack_pred)
        pred_val = model_fusion_tmp.predict(stack_pred_val)
    else:
        #pred_train_sub1 = model_fusion.predict(stack_pred).ravel()
        pred_val = model_fusion_tmp.predict(stack_pred_val).ravel()

    score_1, score_2 = eval_pred(np_val_true_01s, pred_val, str_type)
    if str_type == "EXP":
        score = 0.67*score_1 + 0.33*score_2
    else:
        score = score_1
    log5 = "ens single, score: {0}. score1: {1}, score2: {2}".format(score, score_1, score_2)

    log = log1 + "\n" + log2 + "\n" + log3 + "\n" + log4 + "\n" + log5

    file_result = dir_out + 'result_per_time_and_single' + str_type + '.txt'
    with open(file_result, mode='w') as f:
        f.write(log)
    print(log)
    # ************************************************
    
    # final fusion multi

    # stacked predict data: validation
    stack_pred_val = np.column_stack((pred_val_ens_01s, pred_val_ens_06s, pred_val_ens_12s, pred_val_ens_03s,
                                      pred_val_sub1, pred_val_sub2))

    if str_type == "EXP":
        pred_val_fusion = model_fusion.predict(stack_pred_val)
    else:
        pred_val_fusion = model_fusion.predict(stack_pred_val).ravel()

    score_1, score_2 = eval_pred(np_val_true_01s, pred_val_fusion, str_type)

    if str_type == "EXP":
        score = score_1 * 0.67 + score_2 * 0.33
    else:
        score = score_1

    log_end = "Validation Score: {0:.4f}, score_1: {1:.4f}, score_2: {2:.4f}".format(score, score_1, score_2)
    print(log_end)

    file_result = dir_out + 'fusion_multi_' + str_type + '.txt'
    with open(file_result, mode='w') as f:
        f.write(log_end)

    


# In[ ]:




