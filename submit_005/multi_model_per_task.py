#!/usr/bin/env python
# coding: utf-8

# In[19]:


"""
create and evaluate multi time-window model taht is fusion model of multi target models
 * select type: "EXP", "VA_V", "VA_A"
 * select sub1, sub2 type: "EXP", "VA_V", "VA_A"
 * create "fusion" model
 * evaluate validation *note* labels are averaged in 1sec-window
"""
import numpy as np
import pandas as pd
import glob
import re
import sklearn #機械学習のライブラリ
import lightgbm as lgb
from sklearn.metrics import accuracy_score,mean_squared_error,f1_score
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev
import math
from scipy import stats
import pickle
import os


# In[20]:


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


# In[21]:


def data_balance_multi_term(data, str_type):
    """
    EXP:  0,  1,  2,  3,  4,  5,  6
        0.5,  4, 11,  8,  1,  1,  3
    VA:
    A|V->        -1                                +1
    +1.00~+0.75:  1,  1,   3,  15,  10,   3,   1,   1
    +0.75~+0.50:  1,  1,   1,   2,   1,   1,   1,   1
    +0.50~+0.25:  5,  1,   1,   2,   1,   1,   1,   5
    +0.25~+0.00: 10,  5,   1,   1, 0.5,   1,   5,  20
    +0.00~-0.25:  1, 30,  10,   2,   1,  10,   1,   1
    -0.25~-0.50:  1, 20,  40,   2,   1,  20,   1,   1
    -0.50~-0.75:  1,  1,   1,   2,   2,   1,   1,   1
    -0.75~-1.00:  1,  1,   1,   1,   1,   1,   1,   1
    """
    
    if str_type == "EXP":
        data_list = [pd.DataFrame()]*7 # <- dummy
        arr = [0.5,  4, 11,  8,  1,  1,  3]
        for i in range(7):
            data_list[i] = data.loc[data["Neutral-avg_01s"]==i]
            if arr[i] < 1:
                data_list[i] = data_list[i][::2]
            elif arr[i]>=2:                
                data_list[i] = data_list[i].append([data_list[i]]*arr[i],ignore_index=True)
                """
                for n in range(arr[i]):
                    df_tmp1 = data_list[i].shift(n+1).dropna(how='any')
                    df_tmp1.reset_index(drop = True, inplace = True)
                    df_tmp2 = data_list[i].shift(-(n+1)).dropna(how='any')
                    df_tmp2.reset_index(drop = True, inplace = True)
                    df_tmp = (df_tmp1 + df_tmp2)/2
                    df_tmp["Neutral-avg"] = i
                    data_list[i] = data_list[i].append(df_tmp, ignore_index=True)
                """
        #del data_list[0]
        out_data = pd.concat([x for x in data_list])
    
    else:
        data_list = [pd.DataFrame()]*64 # <- dummy
        arr = [1,  1,   3,  15,  10,   3,   1,   1,
               1,  1,   1,   2,   1,   1,   1,   1,
               5,  1,   1,   2,   1,   1,   1,   5,
               10,  5,   1,   1, 0.5,   1,   5,  20,
               1, 30,  10,   2,   1,  10,   1,   1,
               1, 20,  40,   2,   1,  20,   1,   1,
               1,  1,   1,   2,   2,   1,   1,   1,
               1,  1,   1,   1,   1,   1,   1,   1]
        for aro in range(8):
            for val in range(8):
                i = aro*8 + val
                start_a = 1-(aro*0.25)-0.25
                stop_a = 1-(aro*0.25)
                start_v = val*0.25-1
                stop_v = val*0.25-0.75
                data_list[i] = data.loc[(data["valence-avg_01s"]>=start_v)&(data["valence-avg_01s"]<=stop_v)&
                                        (data["arousal-avg_01s"]>=start_a)&(data["arousal-avg_01s"]<=stop_a)]
                if arr[i] < 1:
                    data_list[i] = data_list[i][::2]
                elif arr[i]>=2:
                    data_list[i] = data_list[i].append([data_list[i]]*arr[i],ignore_index=True)
                    """
                    for n in range(arr[i]):
                        df_tmp1 = data_list[i].shift(n+1).dropna(how='any')
                        df_tmp1.reset_index(drop = True, inplace = True)
                        df_tmp2 = data_list[i].shift(-(n+1)).dropna(how='any')
                        df_tmp2.reset_index(drop = True, inplace = True)
                        df_tmp = (df_tmp1 + df_tmp2)/2
                        #print(len(df_tmp))
                        data_list[i] = data_list[i].append(df_tmp, ignore_index=True)
                    """
        out_data = pd.concat([x for x in data_list])
        
    return out_data


# In[22]:


def load_model(file_model):
    with open(file_model, mode='rb') as fp:
        model = pickle.load(fp)
    return model


# In[23]:


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
    


# In[24]:


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

    #print('** np_data_x',np_data_x.shape)
    #print('** np_data_y',np_data_y.shape)

    return np_data_x, np_data_y


# In[25]:


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

    #print('** np_data_x',np_data_x.shape)
    #print('** np_data_y',np_data_y.shape)

    return np_data_x


# In[26]:


# predict
def predict_data(data_train, data_val, models, str_type, window_time, dir_features = None):
    log = "split data to AU ,pose, gaze, openpose"
    print(log)
    train_au, train_pose, train_lmk, train_op, train_rn = split_data(data_train)
    val_au, val_pose, val_lmk, val_op, val_rn = split_data(data_val)
    
    if dir_features != None:
        file_f = dir_features + "features_" + str_type +"_au.csv"
        train_au = substruct_features(train_au, file_f, str_type, window_time)
        val_au = substruct_features(val_au, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_pose.csv"
        train_pose = substruct_features(train_pose, file_f, str_type, window_time)
        val_pose = substruct_features(val_pose, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_lmk.csv"
        train_lmk = substruct_features(train_lmk, file_f, str_type, window_time)
        val_lmk = substruct_features(val_lmk, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_op.csv"
        train_op = substruct_features(train_op, file_f, str_type, window_time)
        val_op = substruct_features(val_op, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_en.csv"
        train_rn = substruct_features(train_rn, file_f, str_type, window_time)
        val_rn = substruct_features(val_rn, file_f, str_type, window_time)
    
    log = "convert data(pandas) to data(numpy) for LightGBM"
    print(log)
    np_train_au_x, np_train_au_y = make_dataset_for_gbm(train_au, str_type)
    np_train_pose_x, np_train_pose_y = make_dataset_for_gbm(train_pose, str_type)
    np_train_lmk_x, np_train_lmk_y = make_dataset_for_gbm(train_lmk, str_type)
    np_train_op_x, np_train_op_y = make_dataset_for_gbm(train_op, str_type)
    np_train_rn_x, np_train_rn_y = make_dataset_for_gbm(train_rn, str_type)
    
    np_val_au_x, np_val_au_y = make_dataset_for_gbm(val_au, str_type)
    np_val_pose_x, np_val_pose_y = make_dataset_for_gbm(val_pose, str_type)
    np_val_lmk_x, np_val_lmk_y = make_dataset_for_gbm(val_lmk, str_type)
    np_val_op_x, np_val_op_y = make_dataset_for_gbm(val_op, str_type)
    np_val_rn_x, np_val_rn_y = make_dataset_for_gbm(val_rn, str_type)

    log = "predict by au, pose, gaze, openpose, GAPP ensemple"
    print(log)
    if str_type == "EXP":
        pred_train_au = models[0].predict(np_train_au_x)
        pred_train_pose = models[1].predict(np_train_pose_x)
        pred_train_lmk = models[2].predict(np_train_lmk_x)
        pred_train_op = models[3].predict(np_train_op_x)
        pred_train_rn = models[4].predict(np_train_rn_x)
        np_train_ens_x = np.column_stack((pred_train_au, pred_train_pose, pred_train_lmk,
                                          pred_train_op, pred_train_rn))
        pred_train_ens = models[5].predict(np_train_ens_x)
        np_train_true = np_train_au_y.ravel()
    else:
        pred_train_au = models[0].predict(np_train_au_x).ravel()
        pred_train_pose = models[1].predict(np_train_pose_x).ravel()
        pred_train_lmk = models[2].predict(np_train_lmk_x).ravel()
        pred_train_op = models[3].predict(np_train_op_x).ravel()
        pred_train_rn = models[4].predict(np_train_rn_x).ravel()
        np_train_ens_x = np.column_stack((pred_train_au, pred_train_pose, pred_train_lmk, 
                                          pred_train_op, pred_train_rn))
        pred_train_ens = models[5].predict(np_train_ens_x).ravel()
        np_train_true = np_train_au_y.ravel()
        
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
    
    return np_train_true, pred_train_ens, np_val_true, pred_val_ens #pd_pred_train, pd_pred_val


# In[27]:


# predict
def predict_data_sub(data_train, data_val, models, str_type, main_type, window_time, dir_features = None):
    log = "split data to AU ,pose, gaze, openpose"
    print(log)
    train_au, train_pose, train_lmk, train_op, train_rn = split_data(data_train)
    val_au, val_pose, val_lmk, val_op, val_rn = split_data(data_val)
    
    if dir_features != None:
        file_f = dir_features + "features_" + str_type +"_au.csv"
        train_au = substruct_features(train_au, file_f, main_type, window_time)
        val_au = substruct_features(val_au, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_pose.csv"
        train_pose = substruct_features(train_pose, file_f, main_type, window_time)
        val_pose = substruct_features(val_pose, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_lmk.csv"
        train_lmk = substruct_features(train_lmk, file_f, main_type, window_time)
        val_lmk = substruct_features(val_lmk, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_op.csv"
        train_op = substruct_features(train_op, file_f, main_type, window_time)
        val_op = substruct_features(val_op, file_f, main_type, window_time)
        file_f = dir_features + "features_" + str_type +"_en.csv"
        train_rn = substruct_features(train_rn, file_f, main_type, window_time)
        val_rn = substruct_features(val_rn, file_f, main_type, window_time)
    
    log = "convert data(pandas) to data(numpy) for LightGBM"
    print(log)
    np_train_au_x = make_dataset_for_gbm_sub(train_au, main_type)
    np_train_pose_x = make_dataset_for_gbm_sub(train_pose, main_type)
    np_train_lmk_x = make_dataset_for_gbm_sub(train_lmk, main_type)
    np_train_op_x = make_dataset_for_gbm_sub(train_op, main_type)
    np_train_rn_x = make_dataset_for_gbm_sub(train_rn, main_type)
    
    np_val_au_x = make_dataset_for_gbm_sub(val_au, main_type)
    np_val_pose_x = make_dataset_for_gbm_sub(val_pose, main_type)
    np_val_lmk_x = make_dataset_for_gbm_sub(val_lmk, main_type)
    np_val_op_x = make_dataset_for_gbm_sub(val_op, main_type)
    np_val_rn_x = make_dataset_for_gbm_sub(val_rn, main_type)

    log = "predict by au, pose, gaze, openpose, GAPP ensemple"
    print(log)
    if str_type == "EXP":
        pred_train_au = models[0].predict(np_train_au_x)
        pred_train_pose = models[1].predict(np_train_pose_x)
        pred_train_lmk = models[2].predict(np_train_lmk_x)
        pred_train_op = models[3].predict(np_train_op_x)
        pred_train_rn = models[4].predict(np_train_rn_x)
        np_train_ens_x = np.column_stack((pred_train_au, pred_train_pose, pred_train_lmk,
                                          pred_train_op, pred_train_rn))
        pred_train_ens = models[5].predict(np_train_ens_x)
    else:
        pred_train_au = models[0].predict(np_train_au_x).ravel()
        pred_train_pose = models[1].predict(np_train_pose_x).ravel()
        pred_train_lmk = models[2].predict(np_train_lmk_x).ravel()
        pred_train_op = models[3].predict(np_train_op_x).ravel()
        pred_train_rn = models[4].predict(np_train_rn_x).ravel()
        np_train_ens_x = np.column_stack((pred_train_au, pred_train_pose, pred_train_lmk, 
                                          pred_train_op, pred_train_rn))
        pred_train_ens = models[5].predict(np_train_ens_x).ravel()
        
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
    
    return pred_train_ens, pred_val_ens #pd_pred_train, pd_pred_val


# In[28]:


def score_CCC_for_lgbm(preds: np.ndarray, data: lgb.Dataset):
    """Calculate CCC"""
    # true data
    y_true = data.get_label()
    # predict data
    y_pred = preds.ravel()
    
    # Calc CCC
    x_mean = y_pred.mean()
    y_mean = y_true.mean()
    sx2 = ((y_pred-x_mean)*(y_pred-x_mean)).mean()
    sy2 = ((y_true-y_mean)*(y_true-y_mean)).mean()
    sxy = ((y_pred-x_mean)*(y_true-y_mean)).mean()
    CCC = (2 * sxy) / (sx2 + sy2 + (x_mean - y_mean) * (x_mean - y_mean))
    #score_acc = (2 * sxy) / (sx2 + sy2 + (x_mean - y_mean) * (x_mean - y_mean))
    mse = mean_squared_error(y_true, y_pred)
    score_CCC = 2*CCC-mse
    
    # name, result, is_higher_better
    return 'score_CCC', score_CCC, True


# In[29]:


def score_EXP_for_lgbm(preds: np.ndarray, data: lgb.Dataset):
    """Calculate score EXP (0.67*F1 + 0.33*acc)"""
    # true data
    y_true = data.get_label()
    # reshape pred
    N_LABELS = 7  # number of labels
    reshaped_preds = preds.reshape(N_LABELS, len(preds) // N_LABELS)
    # 最尤と判断したクラスを選ぶ　
    y_pred = np.argmax(reshaped_preds, axis=0)
    # calc
    score_1 = f1_score(y_true, y_pred, average='macro') # weighted, macro, micro
    score_2 = accuracy_score(y_true, y_pred)
    score_exp = 0.67*score_1 + 0.33*score_2
    # name, result, is_higher_better
    return 'score_exp', score_exp, True


# In[30]:


# create model and training
def create_and_fit_model_GBM(train_x, train_y, validation_x, validation_y, 
                             learning_rate, num_leaves, num_iter, max_depth, bagging_fraction,
                             feature_fraction,min_child_samples,str_type):
    # set training, validation data
    train_data = lgb.Dataset(train_x, label=train_y)
    eval_data = lgb.Dataset(validation_x, label=validation_y, reference= train_data)
    
    # if target is expression, set parameters to learn [0~6] classification
    if str_type == "EXP":
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass', #'multiclass','regression','binary'
            #'metric':'multi_error', #binary_logloss, binary_error, multi_logloss, multi_error
            "metric" : "None",
            'num_leaves':num_leaves,
            'learning_rate':learning_rate,
            'max_depth':max_depth,
            #'num_iterations':num_iter,
            'verbosity': -1,
            'num_class':7,
            'bagging_fraction':bagging_fraction,
            'feature_fraction':feature_fraction,
            'min_child_samples':min_child_samples
        }
        # training and return model and data
        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=eval_data,
            num_boost_round=400,
            verbose_eval=0,
            early_stopping_rounds=20,
            feval=score_EXP_for_lgbm  # <= set custom metric function
        )
    # if target is VA, set parameters to learn -1 ~ 1 regression
    else:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression', #'multiclass','regression','binary'
            #'metric':'mse', #binary_logloss, binary_error
            "metric" : "None",
            'num_leaves':num_leaves,
            'learning_rate':learning_rate,
            'max_depth':max_depth,
            #'num_iterations':num_iter,
            'verbosity': -1,
            'force_row_wise': True,
            'bagging_fraction':bagging_fraction,
            'feature_fraction':feature_fraction,
            'min_child_samples':min_child_samples
        }
        # training and return model and data
        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=eval_data,
            num_boost_round=400,
            verbose_eval=0,
            early_stopping_rounds=20,
            feval=score_CCC_for_lgbm  # <= set custom metric function
        )

    return lgb_model, train_data, eval_data


# In[31]:


# search parameter
def search_paramter(train_x, train_y, val_x, val_y, rate_list, leaf_list, n_iter_list, depth_list, child_list, str_type):
    model = 0
    count = 1
    b_frac = 1
    f_frac = 1
    max_count = len(rate_list)*len(leaf_list)*len(n_iter_list)*len(depth_list)*len(child_list)
    
    best_score = -1.000
    best_socre_1 = -1.000
    best_socre_2 = -1.000
    best_rate = 0.1
    best_leaf = 16
    best_iter = 100
    best_depth = 8
    best_child = 20
    
    for rate in rate_list:
        for leaf in leaf_list:
            for n_iter in n_iter_list:
                for depth in depth_list:
                    for child in child_list:
                        
                        model, train, validation = create_and_fit_model_GBM(train_x,train_y,val_x,val_y,
                                                                        rate,leaf,n_iter,depth,
                                                                        b_frac,f_frac,child,
                                                                        str_type)
                        
                        if str_type == "EXP":
                            pred = model.predict(val_x)
                            score_1, score_2 = eval_pred(val_y, pred, str_type)
                            score = score_1 * 0.67 + score_2 * 0.33
                        else:
                            pred = model.predict(val_x).ravel()
                            score_1, score_2 = eval_pred(val_y, pred, str_type)
                            score = score_1
                        
                        log = "{0:00004d}/{1:00004d}  Score: {2:.4f}, score_1: {3:.4f},"                        " score_2: {4:.4f},rate: {5:.3f}, leaf: {6}, iter: {7}, depth: {8},"                        " child: {9}".format(count, max_count, score, score_1, score_2,
                                             rate, leaf, n_iter, depth, child)
                        print(log)
                        if score > best_score:
                            best_score = score
                            best_socre_1 = score_1
                            best_socre_2 = score_2
                            best_rate = rate
                            best_leaf = leaf
                            best_iter = n_iter
                            best_depth = depth
                            best_child = child

                        count = count + 1

    log = "Score: {0:.4f}, score_1: {1:.4f}, score_2: {2:.4f}, rate: {3:.3f}, leaf: {4},"    " iter: {5}, depth: {6}, child: {7}".format(best_score, best_socre_1, best_socre_2,
                                                best_rate, best_leaf, best_iter, 
                                                best_depth, best_child
                                               )

    print(log)
    
    return best_score, best_socre_1, best_socre_2, best_rate, best_leaf, best_iter, best_depth, best_child


# In[32]:


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


# In[33]:


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


# In[34]:


def generate_sub_model(train_x, train_y, val_x, val_y, str_type):
    # search parameter

    # initialize
    model = 0
    count = 1
    b_frac = 1
    f_frac = 1
    
    # first loop
    rate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    leaf_list = [8, 12, 16, 24, 32, 40]
    n_iter_list = [100]
    depth_list = [ 4, 8, 12, 16]
    child_list = [5, 15, 30]

    # search param
    score, score_1, socre_2, rate, leaf, n_iter, depth, child = search_paramter(train_x, train_y, 
                                                            val_x, val_y, rate_list, leaf_list, 
                                                            n_iter_list, depth_list, child_list, str_type)

    # next loop
    rate_list = [rate-0.1, rate, rate+0.1]
    leaf_list = [leaf-2, leaf, leaf+2]
    n_iter_list = [n_iter]
    depth_list = [ depth-2, depth, depth+2]
    child_list = [5, 15, 30]

    # search param
    score, score_1, socre_2, rate, leaf, n_iter, depth, child = search_paramter(train_x, train_y, 
                                                            val_x, val_y, rate_list, leaf_list, 
                                                            n_iter_list, depth_list, child_list, str_type)

    log = "Score: {0:.4f}, score_1: {1:.4f}, score_2: {2:.4f}, rate: {3:.3f}, leaf: {4},"    " iter: {5}, depth: {6}, child: {7}".format(score, score_1, socre_2, rate, leaf, 
                                                n_iter, depth, child )

    best_rate = rate
    best_leaf = leaf
    best_iter = n_iter
    best_depth = depth
    best_child = child

    print(log)

    # result:

    model, train, validation = create_and_fit_model_GBM(train_x,train_y, val_x, val_y,
                                                        best_rate, best_leaf, best_iter, best_depth,
                                                        b_frac, f_frac, best_child,
                                                        str_type)

    # calc score and print parameter
    if str_type == "EXP":
        pred = model.predict(val_x) # 7-columns
        score_1, score_2 = eval_pred(val_y, pred, str_type)
        score = score_1 * 0.67 + score_2 * 0.33
    else:
        pred = model.predict(val_x).ravel() # 1-columns to list
        score_1, score_2 = eval_pred(val_y, pred, str_type)
        score = score_1

    log_res = "Score: {0:.4f}, score_1: {1:.4f}, score_2: {2:.4f}, rate: {3:.3f}, leaf: {4}, iter: {5},"    " depth: {6}, child: {7}".format(score, score_1, score_2, rate, leaf,n_iter, depth, child)
    print(log)

    # self predict
    #proba_lmk_train = model.predict_proba(train_x)
    pred_train = model.predict(train_x)

    # val predict
    #proba_lmk_val = model.predict_proba(val_x)
    pred_val = model.predict(val_x)
    
    return model, pred_train, pred_val, log_res


# In[35]:


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


# In[36]:


def generate_multi_task_model(dir_data_train, dir_data_val, 
                               dir_model_01s, dir_model_06s, dir_model_12s, dir_model_fusion,
                               dir_out, target, target_sub1, target_sub2, balance=False, dir_features=None):
    str_type = target
    str_type_sub1 = target_sub1
    str_type_sub2 = target_sub2
    # set output file footer
    str_footer = "fusion_multi_" + str_type
    log = "Target type: {0}".format(str_type)
    print(log)
    
    # read models
    model_01s = read_models(dir_model_01s, str_type, 1)
    model_06s = read_models(dir_model_06s, str_type, 6)
    model_12s = read_models(dir_model_12s, str_type, 12)
    
    model_01s_sub1 = read_models(dir_model_01s, str_type_sub1, 1)
    model_06s_sub1 = read_models(dir_model_06s, str_type_sub1, 6)
    model_12s_sub1 = read_models(dir_model_12s, str_type_sub1, 12)
    
    model_01s_sub2 = read_models(dir_model_01s, str_type_sub2, 1)
    model_06s_sub2 = read_models(dir_model_06s, str_type_sub2, 6)
    model_12s_sub2 = read_models(dir_model_12s, str_type_sub2, 12)
    
    model_fusion = load_model(dir_model_fusion + "model_fusion_single_" + str_type + ".pickle")
    model_fusion_sub1 = load_model(dir_model_fusion + "model_fusion_single_" + str_type_sub1 + ".pickle")
    model_fusion_sub2 = load_model(dir_model_fusion + "model_fusion_single_" + str_type_sub2 + ".pickle")
    
    # search files of training data
    file_train = dir_data_train + "*_01s.h5"
    files_train_01s = [
        filename for filename in sorted(glob.glob(file_train))
    ]
    log = "file number of train 01s: {0}".format(len(files_train_01s))
    print(log)

    file_train = dir_data_train + "*_06s.h5"
    files_train_06s = [
        filename for filename in sorted(glob.glob(file_train))
    ]
    log = "file number of train 06s: {0}".format(len(files_train_06s))
    print(log)

    file_train = dir_data_train + "*_12s.h5"
    files_train_12s = [
        filename for filename in sorted(glob.glob(file_train))
    ]
    log = "file number of train 12s: {0}".format(len(files_train_12s))
    print(log)


    # search files of validation data
    file_val = dir_data_val + "*_01s.h5"
    files_val_01s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 01s: {0}".format(len(files_val_01s))
    print(log)

    file_val = dir_data_val + "*_06s.h5"
    files_val_06s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 06s: {0}".format(len(files_val_06s))
    print(log)

    file_val = dir_data_val + "*_12s.h5"
    files_val_12s = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of val 12s: {0}".format(len(files_val_12s))
    print(log)

    # create base dataset
    log = "data loading...."
    print(log)

    str_time = "_01s"
    #data_train = pd.read_hdf(file_train, 'key')
    data_train_01s = crate_base_data(files_train_01s, str_type, str_time)
    log = "data training 01s shape: {0}".format(data_train_01s.shape)
    print(log)
    data_val_01s = crate_base_data(files_val_01s, str_type, str_time)
    log = "data validation 01s shape: {0}".format(data_val_01s.shape)
    print(log)

    str_time = "_06s"
    data_train_06s = crate_base_data(files_train_06s, str_type, str_time)
    log = "data training 06s shape: {0}".format(data_train_06s.shape)
    print(log)
    data_val_06s = crate_base_data(files_val_06s, str_type, str_time)
    log = "data validation 06s shape: {0}".format(data_val_06s.shape)
    print(log)

    str_time = "_12s"
    data_train_12s = crate_base_data(files_train_12s, str_type, str_time)
    log = "data training 12s shape: {0}".format(data_train_12s.shape)
    print(log)
    data_val_12s = crate_base_data(files_val_12s, str_type, str_time)
    log = "data validation 12s shape: {0}".format(data_val_12s.shape)
    print(log)

    # adjust data shape (same frame)
    log = "adjust data shape (same frame)"
    print(log)
    log = "train data shape) 01s: {0}, 06s: {1}, 12s: {2}".format(data_train_01s.shape, data_train_06s.shape, data_train_12s.shape)
    print(log)
    log = "val data shape) 01s: {0}, 06s: {1}, 12s: {2}".format(data_val_01s.shape, data_val_06s.shape, data_val_12s.shape)
    print(log)

    length_columns = len(data_train_01s.columns)
    base_columns = data_train_01s.columns

    data_train_01s.columns = data_train_01s.columns + "_01s"
    data_train_06s.columns = data_train_06s.columns + "_06s"
    data_train_12s.columns = data_train_12s.columns + "_12s"

    data_val_01s.columns = data_val_01s.columns + "_01s"
    data_val_06s.columns = data_val_06s.columns + "_06s"
    data_val_12s.columns = data_val_12s.columns + "_12s"

    data_train = pd.concat([data_train_01s, data_train_06s, data_train_12s], axis=1)
    if str_type == "EXP":
        data_train = data_train.loc[data_train["Neutral-avg_01s"]>=0]
        data_train = data_train.loc[data_train["Neutral-avg_06s"]>=0]
        data_train = data_train.loc[data_train["Neutral-avg_12s"]>=0]
    data_train = data_train.dropna(how='any')
    train_index = data_train.index

    data_val = pd.concat([data_val_01s, data_val_06s, data_val_12s], axis=1)
    if str_type == "EXP":
        data_val = data_val.loc[data_val["Neutral-avg_01s"]>=0]
        data_val = data_val.loc[data_val["Neutral-avg_06s"]>=0]
        data_val = data_val.loc[data_val["Neutral-avg_12s"]>=0]
    data_val = data_val.dropna(how='any')
    val_index = data_val.index
    
    if balance == True:
        data_train = data_balance_multi_term(data_train, str_type)

    data_train_01s = data_train.iloc[:,0:length_columns]
    data_train_06s = data_train.iloc[:,length_columns:length_columns*2]
    data_train_12s = data_train.iloc[:,length_columns*2:length_columns*3]

    data_val_01s = data_val.iloc[:,0:length_columns]
    data_val_06s = data_val.iloc[:,length_columns:length_columns*2]
    data_val_12s = data_val.iloc[:,length_columns*2:length_columns*3]

    data_train_01s.columns = base_columns
    data_train_06s.columns = base_columns
    data_train_12s.columns = base_columns

    data_val_01s.columns = base_columns
    data_val_06s.columns = base_columns
    data_val_12s.columns = base_columns

    log = "train data shape) 01s: {0}, 06s: {1}, 12s: {2}".format(data_train_01s.shape, data_train_06s.shape, data_train_12s.shape)
    print(log)
    log = "val data shape) 01s: {0}, 06s: {1}, 12s: {2}".format(data_val_01s.shape, data_val_06s.shape, data_val_12s.shape)
    print(log)
    
    # main
    # 01s
    window_time = 1
    #pred_train_01s, pred_val_01s = predict_data(data_train_01s, data_val_01s, model_01s, str_type)
    np_train_true_01s, pred_train_ens_01s, np_val_true_01s, pred_val_ens_01s = predict_data(data_train_01s, data_val_01s,
                                                                                            model_01s, str_type,
                                                                                            window_time, dir_features)
    log = "01s pred shape) train: {0}, val: {1}".format(pred_train_ens_01s.shape, pred_val_ens_01s.shape)
    print(log)

    # 06s
    window_time = 6
    #pred_train_06s, pred_val_06s = predict_data(data_train_06s, data_val_06s, model_06s, str_type)
    np_train_true_06s, pred_train_ens_06s, np_val_true_06s, pred_val_ens_06s = predict_data(data_train_06s, data_val_06s,
                                                                                            model_06s, str_type,
                                                                                            window_time, dir_features)
    log = "06s pred shape) train: {0}, val: {1}".format(pred_train_ens_06s.shape, pred_val_ens_06s.shape)
    print(log)

    # 12s
    window_time = 12
    #pred_train_12s, pred_val_12s = predict_data(data_train_12s, data_val_12s, model_12s, str_type)
    np_train_true_12s, pred_train_ens_12s, np_val_true_12s, pred_val_ens_12s = predict_data(data_train_12s, data_val_12s,
                                                                                            model_12s, str_type,
                                                                                            window_time, dir_features)
    log = "12s pred shape) train: {0}, val: {1}".format(pred_train_ens_12s.shape, pred_val_ens_12s.shape)
    print(log)
    
    # sub1 (with fusion single)
    # 01s
    window_time = 1
    #pred_train_01s, pred_val_01s = predict_data(data_train_01s, data_val_01s, model_01s, str_type)
    pred_train_ens_01s_sub1, pred_val_ens_01s_sub1 = predict_data_sub(data_train_01s, data_val_01s,
                                                                      model_01s_sub1, str_type_sub1, str_type,
                                                                      window_time, dir_features)
    log = "01s pred shape) train: {0}, val: {1}".format(pred_train_ens_01s_sub1.shape, pred_val_ens_01s_sub1.shape)
    print(log)

    # 06s
    window_time = 6
    pred_train_ens_06s_sub1, pred_val_ens_06s_sub1 = predict_data_sub(data_train_06s, data_val_06s,
                                                                      model_06s_sub1, str_type_sub1, str_type,
                                                                      window_time, dir_features)
    log = "06s pred shape) train: {0}, val: {1}".format(pred_train_ens_06s_sub1.shape, pred_val_ens_06s_sub1.shape)
    print(log)

    # 12s
    window_time = 12
    pred_train_ens_12s_sub1, pred_val_ens_12s_sub1 = predict_data_sub(data_train_12s, data_val_12s,
                                                                      model_12s_sub1, str_type_sub1, str_type,
                                                                      window_time, dir_features)
    log = "12s pred shape) train: {0}, val: {1}".format(pred_train_ens_12s_sub1.shape, pred_val_ens_12s_sub1.shape)
    print(log)

    # fusion sub1
    # stacked predict data: train
    stack_pred = np.column_stack((pred_train_ens_01s_sub1, pred_train_ens_06s_sub1, pred_train_ens_12s_sub1))

    # stacked predict data: validation
    stack_pred_val = np.column_stack((pred_val_ens_01s_sub1, pred_val_ens_06s_sub1, pred_val_ens_12s_sub1))

    if str_type_sub1 == "EXP":
        pred_train_sub1 = model_fusion_sub1.predict(stack_pred)
        pred_val_sub1 = model_fusion_sub1.predict(stack_pred_val)
    else:
        pred_train_sub1 = model_fusion_sub1.predict(stack_pred).ravel()
        pred_val_sub1 = model_fusion_sub1.predict(stack_pred_val).ravel()

    # sub2 (with fusion single)
    # 01s
    window_time = 1
    #pred_train_01s, pred_val_01s = predict_data(data_train_01s, data_val_01s, model_01s, str_type)
    pred_train_ens_01s_sub2, pred_val_ens_01s_sub2 = predict_data_sub(data_train_01s, data_val_01s,
                                                                      model_01s_sub2, str_type_sub2, str_type,
                                                                      window_time, dir_features)
    log = "01s pred shape) train: {0}, val: {1}".format(pred_train_ens_01s_sub2.shape, pred_val_ens_01s_sub2.shape)
    print(log)

    # 06s
    window_time = 6
    pred_train_ens_06s_sub2, pred_val_ens_06s_sub2 = predict_data_sub(data_train_06s, data_val_06s,
                                                                      model_06s_sub2, str_type_sub2, str_type,
                                                                      window_time, dir_features)
    log = "06s pred shape) train: {0}, val: {1}".format(pred_train_ens_06s_sub2.shape, pred_val_ens_06s_sub2.shape)
    print(log)

    # 12s
    window_time = 12
    pred_train_ens_12s_sub2, pred_val_ens_12s_sub2 = predict_data_sub(data_train_12s, data_val_12s,
                                                                      model_12s_sub2, str_type_sub2, str_type,
                                                                      window_time, dir_features)
    log = "12s pred shape) train: {0}, val: {1}".format(pred_train_ens_12s_sub2.shape, pred_val_ens_12s_sub2.shape)
    print(log)

    # fusion sub2
    # stacked predict data: train
    stack_pred = np.column_stack((pred_train_ens_01s_sub2, pred_train_ens_06s_sub2, pred_train_ens_12s_sub2))

    # stacked predict data: validation
    stack_pred_val = np.column_stack((pred_val_ens_01s_sub2, pred_val_ens_06s_sub2, pred_val_ens_12s_sub2))

    if str_type_sub2 == "EXP":
        pred_train_sub2 = model_fusion_sub2.predict(stack_pred)
        pred_val_sub2 = model_fusion_sub2.predict(stack_pred_val)
    else:
        pred_train_sub2 = model_fusion_sub2.predict(stack_pred).ravel()
        pred_val_sub2 = model_fusion_sub2.predict(stack_pred_val).ravel()

    # ensemble    
    # stacked predict data: train
    stack_pred = np.column_stack((pred_train_ens_01s, pred_train_ens_06s, pred_train_ens_12s,
                                  pred_train_sub1, pred_train_sub2))

    # stacked predict data: validation
    stack_pred_val = np.column_stack((pred_val_ens_01s, pred_val_ens_06s, pred_val_ens_12s,
                                      pred_val_sub1, pred_val_sub2))
    
    # generate single-term ensemble model
    model_ens, pred_train_ens, pred_val_ens, log_res_ens = generate_sub_model(stack_pred, np_train_true_01s, 
                                                                              stack_pred_val, np_val_true_01s,
                                                                              str_type)

    # save model
    f = dir_out + 'model_' + str_footer + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_ens, fp)
    
    #save log
    print(log_res_ens)

    file_result = dir_out + 'result_' + str_footer + '.txt'
    with open(file_result, mode='w') as f:
        f.write(log_res_ens)

        
    log = "*** FINISHED *** Target type: {0}".format(str_type)
    print(log)
        
    


# In[ ]:




