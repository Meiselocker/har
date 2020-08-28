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
mhealth = Har.Har(sf = 50, wl = 5, ol = 2.5, resample = 1)  
   
############Data specification################

colnames = ["A_chest_1", "A_chest_2", "A_chest_3", 
     "ECG_1", "ECG_2",
     "A_ankle_1", "A_ankle_2", "A_ankle_3",
     "G_ankle_1", "G_ankle_2", "G_ankle_3",
     "M_ankle_1", "M_ankle_2", "M_ankle_3",
     "A_wirst_1", "A_wrist_2", "A_wrist_3",
     "G_wrist_1", "G_wrist_2", "G_wrist_3",
     "M_wrist_1","M_wrist_2", "M_wrist_3", "label"]

used_colnames = ["A_chest_1", "A_chest_2", "A_chest_3",
     "A_ankle_1", "A_ankle_2", "A_ankle_3",
     "G_ankle_1", "G_ankle_2", "G_ankle_3",
     "M_ankle_1", "M_ankle_2", "M_ankle_3",
     "A_wirst_1", "A_wrist_2", "A_wrist_3",
     "G_wrist_1", "G_wrist_2", "G_wrist_3",
     "M_wrist_1","M_wrist_2", "M_wrist_3"]

a_cols = ["A_" in x for x in used_colnames]    

##########Reading and processing data #############
print("Reading and processing data...")
for i in range(1,11):
    path = os.path.join(script_dir, "mhealth/mHealth_subject{}.log".format(str(i)))
    mhealth.read_signal(path, colnames = colnames, use_cols = used_colnames,
                       sep = "\t", ID = str(i))    
    mhealth.process_signal(ACDC_on = a_cols, ID = str(i))
    mhealth.read_labels(path, colnames = colnames, label_col = "label", sep ="\t",
                       ID = str(i))  
print("Reading and processing completed")

#########Removing Null labels#################################

mhealth.remove_labels([0])

#########Training, recognizing activity and testing#########
clf = LogisticRegression(n_jobs = -1, max_iter = 1000)
acc_scores = []
for i in range(1,11):
    test_ID = str(i)
    train_IDs = [str(i) for i in range(1,11) if str(i)!=test_ID]
    mhealth.learn(clf, train_IDs)
    mhealth.predict(test_ID)
    pred_labs = mhealth.predicted_labels[test_ID]['predicted_label'].values
    labs = mhealth.labels[test_ID]['label'].values
    score = np.mean(pred_labs == labs)
    acc_scores.append(score)
    print("Accuracy score on individual # {} is {}".format(i, score))
print("Average accuracy score is {}".format(np.mean(acc_scores)))




