# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:29:32 2020

@author: debac
"""

############Importing class and instantiating har object############
from har import Har, funs
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
script_dir = os.path.dirname(__file__)

#Instantiate a har object, specifying sampling frequency (sf), window length
#(wl, in seconds), overlap (ol, in seconds), and resampling rate (resample).
dailac = Har.Har(sf = 204.8, wl = 5, ol = 2.5, resample = 4)  
   
############Data specification################

#colnames of the raw wearables' data
colnames = ['A_wrist_1','A_wrist_2','A_wrist_3','G_wrist_1','G_wrist_2',
            'G_wrist_3','A_chest_1','A_chest_2','A_chest_3','G_chest_1',
            'G_chest_2','G_chest_3','A_hip_1','A_hip_2','A_hip_3','G_hip_1',
            'G_hip_2','G_hip_3','A_ankle_1','A_ankle_2','A_ankle_3',
            'G_ankle_1','G_ankle_2','G_ankle_3','label']
#acceleration columns on which AC/DC filtering will be performed
a_cols = ["A_" in x for x in colnames[:-1]]    

##########Reading and processing data #############
print("Reading and processing data...")
for i in range(1,20):#reading data for each ID
    path = os.path.join(script_dir, "dailac/dataset_{}.txt".format(str(i)))
    # reading all columns but the last one (labels)
    dailac.read_signal(path, colnames = colnames, use_cols = colnames[:-1],
                       ID = str(i))
    #AC/DC filtering is performed on acceleration columns only
    dailac.process_signal(ACDC_on = a_cols, ID = str(i))
    dailac.read_labels(path, colnames = colnames, label_col = "label",
                       ID = str(i))  
print("Reading and processing completed")

#########Training, recognizing activity and testing#########

#hierarchical structures of activity for multi-level classification
meta_classes = [[1], [2], [3,4], [5,6], [7,8,9], [10], [11,12], [13]]
#creates function generating a LR classifier
my_clf = lambda: LogisticRegression(n_jobs = -1, max_iter = 1000)

print("Training on 18 and testing on 1...")
acc_scores = []
for i in range(1,20):
    test_ID = str(i)
    train_IDs = [str(i) for i in range(1,20) if str(i)!=test_ID]
    dailac.hierarchical_learn(my_clf, meta_classes, train_IDs)
    dailac.hierarchical_predict(test_ID)
    pred_labs = dailac.predicted_labels[test_ID]['predicted_label'].values
    labs = dailac.labels[test_ID]['label'].values
    score = np.mean(pred_labs == labs)#accuracy
    acc_scores.append(score)
    print("Accuracy score on individual # {} is {}".format(i, score))

print("Average accuracy score is {}".format(np.mean(acc_scores)))

dailac.save_model("model.pkl")
