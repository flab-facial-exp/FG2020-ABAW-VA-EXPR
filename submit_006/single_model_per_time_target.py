#!/usr/bin/env python
# coding: utf-8

# In[15]:


"""
create and evaluate single time-window model and single target
 * select type: "EXP", "VA_V", "VA_A"
 * select time-window: "_01s", "_06s", "_12s"
 * create "au", "pose", "gaze", "openpose" and "ensemble" model
 * evaluate validation *note* labels are averaged in time-window
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


# In[16]:


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


# In[17]:


def data_balance(data, str_type):
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
            data_list[i] = data.loc[data["Neutral-avg"]==i]
            if arr[i] < 1:
                data_list[i] = data_list[i][::2]
            elif arr[i]>=2:                
                data_list[i] = data_list[i].append([data_list[i]]*arr[i],ignore_index=True)
                
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
                data_list[i] = data.loc[(data["valence-avg"]>=start_v)&(data["valence-avg"]<=stop_v)&
                                        (data["arousal-avg"]>=start_a)&(data["arousal-avg"]<=stop_a)]
                if arr[i] < 1:
                    data_list[i] = data_list[i][::2]
                elif arr[i]>=2:
                    data_list[i] = data_list[i].append([data_list[i]]*arr[i],ignore_index=True)
                    
        out_data = pd.concat([x for x in data_list])
        
    return out_data


# In[18]:


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
    


# In[19]:


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

    print('** np_data_x',np_data_x.shape)
    print('** np_data_y',np_data_y.shape)

    return np_data_x, np_data_y


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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
    
    """
    # save model
    f = dir_out + 'model_rn' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model, fp)
    #

    # self predict
    #proba_lmk_train = model.predict_proba(train_x)
    pred_rn_train = model.predict(train_x)

    # val predict
    #proba_lmk_val = model.predict_proba(val_x)
    pred_rn_val = model.predict(val_x)
    """


# In[26]:


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


# In[27]:


def generate_single_model(dir_train, dir_val, dir_out, target, window_time, balance = True, dir_features = None):
    str_type = target
    str_time = "_{0}s".format(str(window_time).zfill(2))
    str_footer = "_gbm_" + str_type
    log = "Target type: {0}, time: {1}s".format(str_type, window_time)
    print(log)
    
    # search files of training data
    file_train = dir_train + "*" + str_time + ".h5"
    files_train = [
        filename for filename in sorted(glob.glob(file_train))
    ]
    log = "file number of training: {0}".format(len(files_train))
    print(log)
    
    # search files of validation data
    file_val = dir_val + "*" + str_time + ".h5"
    files_val = [
        filename for filename in sorted(glob.glob(file_val))
    ]
    log = "file number of validation: {0}".format(len(files_val))
    print(log)

    # create base dataset
    log = "data loading...."
    print(log)
    #data_train = pd.read_hdf(file_train, 'key')
    data_train = crate_base_data(files_train, str_type, str_time)
    log = "data training shape: {0}".format(data_train.shape)
    print(log)
    #data_val = pd.read_hdf(file_val, 'key')
    data_val = crate_base_data(files_val, str_type, str_time)
    log = "data validation shape: {0}".format(data_val.shape)
    print(log)
    
    # "change EXP:-1 to 0"
    if str_type == "EXP":
        # cut EXP:-1
        data_train2 = data_train[data_train['Neutral-avg'] >= 0]
        data_val2 = data_val[data_val['Neutral-avg'] >= 0]
    else:
        data_train2 = data_train
        data_val2 = data_val
    # "cut nan"
    log = " src shape, train: {0}, validation: {1}".format(data_train2.shape, data_val2.shape)
    print(log)
    data_train2.dropna(how='any', inplace=True)
    data_val2.dropna(how='any', inplace=True)
    log = "dist shape, train: {0}, validation: {1}".format(data_train2.shape, data_val2.shape)
    print(log)
    
    if balance == True:
        log = "data shape before balance: {0}".format(data_train2.shape)
        print(log)
        data_train2 = data_balance(data_train2, str_type)
        log = "data shape after balance: {0}".format(data_train2.shape)
        print(log)
        
    # reset index
    data_train2 = data_train2.reset_index(drop=True)
    data_val2 = data_val2.reset_index(drop=True)
    
    # split data to AU ,pose, gaze, openpose
    # train
    data_au_train, data_pose_train, data_lmk_train, data_op_train, data_rn_train = split_data(data_train2)
    # validation
    data_au_val, data_pose_val, data_lmk_val, data_op_val, data_rn_val = split_data(data_val2)
    
    # str_type
    if dir_features != None:
        file_f = dir_features + "features_" + str_type +"_au.csv"
        data_au_train = substruct_features(data_au_train, file_f, str_type, window_time)
        data_au_val = substruct_features(data_au_val, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_pose.csv"
        data_pose_train = substruct_features(data_pose_train, file_f, str_type, window_time)
        data_pose_val = substruct_features(data_pose_val, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_lmk.csv"
        data_lmk_train = substruct_features(data_lmk_train, file_f, str_type, window_time)
        data_lmk_val = substruct_features(data_lmk_val, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_op.csv"
        data_op_train = substruct_features(data_op_train, file_f, str_type, window_time)
        data_op_val = substruct_features(data_op_val, file_f, str_type, window_time)
        file_f = dir_features + "features_" + str_type +"_en.csv"
        data_rn_train = substruct_features(data_rn_train, file_f, str_type, window_time)
        data_rn_val = substruct_features(data_rn_val, file_f, str_type, window_time)
        
    
    # convert data(pandas) to time domain data(numpy) for lstm
    ### *_y, *_group is the same value
    # data au
    np_au_x, np_au_y = make_dataset_for_gbm(data_au_train, str_type)
    # data pose
    np_pose_x, np_pose_y = make_dataset_for_gbm(data_pose_train, str_type)
    # data lmk pca
    np_lmk_x, np_lmk_y = make_dataset_for_gbm(data_lmk_train, str_type)
    # data openpose
    np_op_x, np_op_y = make_dataset_for_gbm(data_op_train, str_type)
    # data resnet
    np_rn_x, np_rn_y = make_dataset_for_gbm(data_rn_train, str_type)

    # data au
    np_au_x_val, np_au_y_val = make_dataset_for_gbm(data_au_val, str_type)
    # data pose
    np_pose_x_val, np_pose_y_val = make_dataset_for_gbm(data_pose_val, str_type)
    # data lmk pca
    np_lmk_x_val, np_lmk_y_val = make_dataset_for_gbm(data_lmk_val, str_type)
    # data openpose
    np_op_x_val, np_op_y_val = make_dataset_for_gbm(data_op_val, str_type)
    # data resnet
    np_rn_x_val, np_rn_y_val = make_dataset_for_gbm(data_rn_val, str_type)
    
    log = " ***** finished Preprocessing ***** "
    print(log)
    
    # au
    train_x = np_au_x
    train_y = np_au_y.ravel()
    val_x = np_au_x_val
    val_y = np_au_y_val.ravel()
    model_au, pred_train_au, pred_val_au, log_res_au = generate_sub_model(train_x, train_y, val_x, val_y, str_type)
    
    # head
    train_x = np_pose_x
    train_y = np_pose_y.ravel()
    val_x = np_pose_x_val
    val_y = np_pose_y_val.ravel()
    model_pose, pred_train_pose, pred_val_pose, log_res_pose = generate_sub_model(train_x, train_y, val_x, val_y, str_type)
    
    # gaze
    train_x = np_lmk_x
    train_y = np_lmk_y.ravel()
    val_x = np_lmk_x_val
    val_y = np_lmk_y_val.ravel()
    model_lmk, pred_train_lmk, pred_val_lmk, log_res_lmk = generate_sub_model(train_x, train_y, val_x, val_y, str_type)
    
    # openpose
    train_x = np_op_x
    train_y = np_op_y.ravel()
    val_x = np_op_x_val
    val_y = np_op_y_val.ravel()
    model_op, pred_train_op, pred_val_op, log_res_op = generate_sub_model(train_x, train_y, val_x, val_y, str_type)
    
    # resnet
    train_x = np_rn_x
    train_y = np_rn_y.ravel()
    val_x = np_rn_x_val
    val_y = np_rn_y_val.ravel()
    model_rn, pred_train_rn, pred_val_rn, log_res_rn = generate_sub_model(train_x, train_y, val_x, val_y, str_type)
    
    # data set for ensemble
    # stacked predict data: train
    stack_pred = np.column_stack((pred_train_au, pred_train_pose, pred_train_lmk, pred_train_op, pred_train_rn))

    # stacked predict data: validation
    stack_pred_val = np.column_stack((pred_val_au, pred_val_pose, pred_val_lmk, pred_val_op, pred_val_rn))

    # generate single-term ensemble model
    model_ens, pred_train_ens, pred_val_ens, log_res_ens = generate_sub_model(stack_pred, train_y, 
                                                                              stack_pred_val, val_y,
                                                                              str_type)

    # save models
    # save model
    f = dir_out + 'model_au' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_au, fp)
    f = dir_out + 'model_pose' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_pose, fp)
    f = dir_out + 'model_lmk' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_lmk, fp)
    f = dir_out + 'model_op' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_op, fp)
    f = dir_out + 'model_rn' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_rn, fp)
    f = dir_out + 'model_ens' + str_footer + str_time + '.pickle'
    with open(f, mode='wb') as fp:
        pickle.dump(model_ens, fp)
    
    #save log
    log_end = "<au>: {0}\n<head>: {1}\n<gaze>: {2}\n<pose>: {3}\n<rn>: {4}\n<ens>: {5}".format(log_res_au, 
                                                    log_res_pose, log_res_lmk, log_res_op, log_res_rn, log_res_ens)
    print(log_end)

    file_result = dir_out + 'result' + str_footer + str_time + '.txt'
    with open(file_result, mode='w') as f:
        f.write(log_end)
        
    log = "*** FINISHED *** Target type: {0}, time: {1}s".format(str_type, window_time)
    print(log)
    


# In[ ]:




