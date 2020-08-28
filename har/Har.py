# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:55:50 2020

@author: debac
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import mode
from har import funs


class Har:
    def __init__(self, sf, wl, ol = None, resample = None):
        self.sf = sf
        self.wl = int(wl*sf)
        if not ol:
            self.ol = int(wl//2)
        else:
            self.ol = int(ol*sf)
        self.resample = resample
        self.data = {}
        self.feats = {}
        self.labels = {}
        self.predicted_labels = {}
    
    def read_signal(self, path, colnames= None, 
                    use_cols = None, sep = ',', ID = 'NA'):
        """
        Reads signals in from a .CSV file. Data are read and stored as a Pandas
        dataframe in a dictionary object, where the keys are the IDs passed as
        argument. Raw data are available at `self.data`, and are used for feature
        extraction. 

        Parameters
        ----------
        path : STRINGS. 
            Path to the .CSV file.
        colnames : LIST of STRINGS, optional
            Names of signals, to be assigned to the columns. If None is 
            passed, names are inferred from first line.
        use_cols : LIST of STRINGS or INTEGERS, optional.
            Select the columns with signals. The default is None, meaning
            all columns are read.
        sep : STRING, optional
            Separator of the CSV file. The default is ','.
        ID : STR or INT, optional
            The ID with which the signals are associated, e.g. a name, data, 
            file number etc. The default is 'NA', i.e. data are stored under 'NA'.

        Returns
        -------
        None

        """
        data = pd.read_csv(path, sep = sep, names = colnames,
                           usecols = use_cols)
        if self.resample:
            data = data.iloc[::self.resample,:]
            data.reset_index(inplace = True, drop = True)
        if self.data.get(ID):
            self.data[ID] = pd.concat([self.data[ID], data], axis = 0)
        else:
            self.data[ID] = data

    def process_signal(self, ACDC_on = "all", ID = 'NA'):
        """
        Processes previously read in data into temporal and frequential features.
        The process includes filtering of the signal and extraction of AC and
        DC components. The columns on which this should be applied are specified
        in the ACDC_on arguments. The ID for which proecessing should be made 
        is passed as well. 
        Features are stored as a Pandas dataframe in a dictionary by ID, and 
        can be accessed using `self.feats`.

        Parameters
        ----------
        ACDC_on : LIST of INT or STR , optional
            LIST of columns to select signals for which AC DC components should
            be extracted. The default is "all".
        ID : STR or INT, optional.
            For the data of which ID the processing should be made. 

        Returns
        -------
        None

        """
        #filtering to AC DC components
        if ACDC_on =="all":
            AC,DC = funs.get_AC_DC(self.data[ID], self.sf, 1, 2)
        else:
            if isinstance(ACDC_on[0], int):
                X = self.data[ID].iloc[:,ACDC_on]
            else:
                X = self.data[ID].loc[:,ACDC_on]
            AC,DC = funs.get_AC_DC(X, self.sf, 1, 2)        
        data = pd.concat([self.data[ID], AC, DC], axis = 1)
        data_names = data.columns.to_list()
        #scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # extracting temporal and frequential features column wise (i.e. by signal)
        all_temp_feats = []
        all_freq_feats = []
        n_cols = data.shape[1]
        for col in range(n_cols):
            x = funs.reshape_to_windows(data[:,col], self.wl, self.ol)
            temp_feats, temp_feats_names =  funs.extract_temporal_features(x)
            all_temp_feats.append(temp_feats)
            x -= np.mean(x, axis = 1)[:,None]
            ft = np.fft.fft2(x)/self.wl
            freq_feats, freq_feats_names =\
                funs.extract_frequential_features(ft, self.sf)
            all_freq_feats.append(freq_feats)
        #arrays of names for features
        all_temp_feats_names = [col_name+feat_name \
            for col_name in data_names for feat_name in temp_feats_names]
        all_freq_feats_names = [col_name+feat_name \
            for col_name in data_names for feat_name in freq_feats_names]
        #converting to pandas dataframe
        stacked_feats = np.concatenate(all_temp_feats+all_freq_feats,
                                        axis = 1)
        all_names = all_temp_feats_names + all_freq_feats_names
        feats = pd.DataFrame(stacked_feats, columns = all_names)
        self.feats[ID] = feats

    def read_labels(self, path, label_col, colnames = None, 
                 sep = ',', ID = None):
        """
        Reads labels from a column of a .CSV file. Arguments are similar to
        the method `read_signal`. One label per window is sampled (the mode) to
        match the number of features. Labels are stored as a Pandas dataframe
        in a dictionary by ID. 

        Parameters
        ----------
        path : STRINGS. 
            Path to the .CSV file.
        colnames : LIST of STRINGS, optional
            Names of signals, to be assigned to the columns. If None is 
            passed, names are inferred from first line.
        label_col : STR or INT
            In what column of the .CSV labels should be found.
        sep : STRING, optional
            Separator of the CSV file. The default is ','.
        ID : STR or INT, optional
            The ID with which the signals are associated, e.g. a name, data, 
            file number etc. The default is 'NA', i.e. data are stored under 'NA'.
        
        Returns
        -------
        None.

        """
        label_col = [label_col]
        labels = pd.read_csv(path, sep = sep, names = colnames,
                       usecols = label_col)
        if self.resample:
            labels = labels.iloc[::self.resample,:].values[:,0]
            labels = funs.reshape_to_windows(labels, self.wl, self.ol)
            labels = mode(labels, axis = 1)[0][:,0]
            labels = pd.DataFrame(labels, columns = ['label'])
        
        if self.labels.get(ID):
            self.labels[ID] = pd.concat([self.labels[ID], labels], axis = 0)
        else:
            self.labels[ID] = labels

    def learn(self, classifier, train_IDs = "all"):
            """
    
          trains a classifierfor a list of IDs. 
          Classifier is saved in a list under `self.clfs`. 
          
            Parameters
            ----------
            classifier : an instantiated sklearn classifier
            train_IDs : (LIST of) STR or (LIST of) INT, optional
                IDs to be used for training. The default is "all", i.e. all ID's are used.
    
            Returns
            -------
            None.
    
            """
            
            if train_IDs == "all":
                train_IDs = list(self.feats.keys())
            else:
                train_IDs = list(train_IDs)
            X = np.concatenate([self.feats[ID].values for ID in train_IDs],
                               axis = 0)
            y = np.concatenate([self.labels[ID].values[:,0] for ID in train_IDs],
                               axis = 0)

            #scaling X 
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            classifier.fit(X,y)
            self.clfs = [classifier]

    def predict(self, ID):
        """
        Predicts labels from features for an ID using previously trained 
        classifier. Accessible as pandas DataFrame at `self.predicted_labels`

        Parameters
        ----------
        ID : STR or INT
            ID for which labels are to be predicted. .

        Returns
        -------
        None

        """
        X = self.feats[ID].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y_ = self.clfs[0].predict(X)
        self.predicted_labels[ID] = pd.DataFrame(y_,
                                                 columns = ["predicted_label"])
    
    def remove_labels(self, labels):
        """
        Removes irrelevant labels and their corresponding features 
        based on a list of values. Useful before training when data contains
        irrelevant episodes. Should be run after reading and transforming data.
        
        labels: list of INT or STR.
            labels for which data should be deleted.

        Returns
        -------
        None.

        """
        for ID in self.labels.keys():
            mask = self.labels[ID]['label'].isin(labels)
            self.labels[ID] = self.labels[ID][~mask]
            self.feats[ID] = self.feats[ID][~mask]
        
    def hierarchical_learn(self, classifier, meta_classes, train_IDs = "all"):
        """

      returns list of classifiers for hierarchical classification 
      [super classifier, intra-metaclass classifier 1, .., intra-metaclass 
       classifier n], given a function instantiating a classifier, a list of
      classes divided into n meta classes and IDs to train on.
      Classifiers are saved as list under `self.clfs`. 
      
        Parameters
        ----------
        classifier : FUNCTION or LIST of sklearn classifier
            Either a function for generating a sklearn classifier (NOT an 
            instance; i.e. LogisticRegression and not LogisticRegression()), or
            a list of n+1 classifier instances, first a super classifier and then
            n classifiers for the n meta-classes.
        meta_classes : List of lists of INT or STR.
            List of meta-classes listing target classes by name.
        train_IDs : (LIST of) STR or (LIST of) INT, optional
            IDs to be used for training. The default is "all", i.e. all ID's are used.

        Returns
        -------
        None.

        """
        
        if train_IDs == "all":
            train_IDs = list(self.feats.keys())
        else:
            train_IDs = list(train_IDs)
        X = np.concatenate([self.feats[ID].values for ID in train_IDs],
                           axis = 0)
        y = np.concatenate([self.labels[ID].values[:,0] for ID in train_IDs],
                           axis = 0)
        mult_classes = [classes for classes in meta_classes if len(classes) > 1]
        if isinstance(classifier, list):
            clfs = classifier
        else:
            n = len(mult_classes)
            clfs = [classifier() for i in range(n+1)]  
        #scaling X 
        scaler = StandardScaler()
        X_base = scaler.fit_transform(X)

        #if hierarchical learning
        if len(mult_classes)>0:
            map_dict = {i:mc[0] for mc in meta_classes for i in mc}
            y_ = np.array([map_dict[i] for i in y])#converting y to meta-classes
            clfs[0].fit(X_base, y_)
            for clf, classes in zip(clfs[1:], mult_classes):#training each classifier discriminating within meta-class
                ind = np.isin(y, classes)
                scaled_X = scaler.fit_transform(X[ind,:])
                clf.fit(scaled_X, y[ind])
        else:
            clfs[0].fit(X_base, y)
     
        self.clfs = clfs    

    def hierarchical_predict(self, ID):
        """
        Predicts labels from features for an ID using previously trained 
        classifiers. Accessible as pandas DataFrame at `self.predicted_labels`

        Parameters
        ----------
        ID : STR or INT
            ID for which labels are to be predicted. .

        Returns
        -------
        None

        """
        X = self.feats[ID].values
        scaler = StandardScaler()
        X_base = scaler.fit_transform(X)
        y_ = self.clfs[0].predict(X_base)#predicting meta classes (base)
        for clf in self.clfs[1:]:#discriminating within meta-classes
          ind = np.isin(y_, clf.classes_)
          scaler = StandardScaler()
          scaled_X = scaler.fit_transform(X[ind,:])
          y_pred = clf.predict(scaled_X)
          y_[ind] = y_pred
        self.predicted_labels[ID] = pd.DataFrame(y_, columns = ["predicted_label"])

    def save_model(self, path):
        """
        Saves the list fo classifier(s) for later use.

        Parameters
        ----------
        path : STR
            Where model should be saved.

        Returns
        -------
        None.

        """
        import pickle
        with open(path, 'wb') as dst:
            pickle.dump(self.clfs, dst)
    
    def load_model(self, path):
        """
        Loads previously trained model (list of classifier(s)) from path.
        Accessible at `self.clfs`

        Parameters
        ----------
        path : STR
            Path to saved model.

        Returns
        -------
        None.

        """
        import pickle
        with open(path, 'rb') as src:
            self.clfs = pickle.load(src)
