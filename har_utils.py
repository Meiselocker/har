from scipy import signal as sg
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_AC_DC(X, sf, order = 1, crit_freq = 2):
  """
  returns high-pass and low-pass filtered signals using butterworth filter
  X: original signal
  sf = sampling frequency
  order: order of filter
  crit_freq: critical frequency of filter
  """
  sos_low = sg.butter(order, crit_freq, 'lp', fs = sf, output = 'sos')
  sos_high = sg.butter(order, crit_freq, 'hp', fs = sf, output = 'sos')
  AC = np.zeros(X.shape)
  DC = np.zeros(X.shape)
  for col in range(X.shape[1]):
    AC [:,col] = sg.sosfilt(sos_low, X[:,col])
    DC [:,col] = sg.sosfilt(sos_high, X[:,col])
  return pd.DataFrame(AC), pd.DataFrame(DC)

def reshape_to_windows(x, wl, ol):
  """
  Segments signal into windows by reshaping it into 2D
  x: 1-d array to reshape
  wl: no. of samples in window
  ol: no. of samples overlap
  """
  assert wl>ol, 'Window must be longer than overlap'
  step=int(wl-ol)
  nrows = int(1+(x.size-wl)//step)
  n = int(x.strides[0])
  return np.lib.stride_tricks.as_strided(x, shape=(nrows,int(wl)),
                                        strides=(step*n,n))

def extract_temporal_features(X):
  """
  exctracting time domain feature from windowed raw signal (2D)
  X: 2D windowed signal
  """
  m = np.mean(X, axis = 1)
  sd = np.std(X, axis = 1)
  kurt = kurtosis(X, axis = 1)
  sk = skew(X, axis = 1)
  rms = np.sqrt(np.mean(X**2, axis = 1))
  zc = np.sum(np.diff(X>=m[:,None], axis = 1), axis = 1)
  q = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])
  Q = np.quantile(X, q, axis=1)
  range_ = Q[-1,:]-Q[0,:]
  df = np.concatenate([m[:,None],sd[:,None],kurt[:,None],sk[:,None],
                       rms[:,None],zc[:,None],Q.T,range_[:,None]], axis = 1)
  fun_names = ["_mean", "_sd", "_kurtosis", "_skew", "_rms", "_zero_crossing",
  "_min","_q5", "_q10", "_q20", "_q30", "_q40", "_q50",
  "_q60", "_q70", "_q80", "_q90", "_q95", "_max", "_range"]                   
  return df, fun_names


def extract_frequential_features (FT, sf):
  """
  exctracting frequncy domain feature from windowed Fourier transform (2D)
  X: 2D windowed signal
  sf: sampling frequency
  """
  sf = int(sf)
  #dimensions
  m = FT.shape[0]
  n = FT.shape[1]
  #discarding mirror part
  FT = FT[:,:n//2]
  #frequencies of the transofm
  freqs = np.fft.fftfreq(n, 1/sf)[1:n//2]
  #the spectral density is the squared of the absolute
  Spec = np.abs(FT)**2
  #Energy
  E = np.sum(Spec, axis=1)/(n//2)
  #density
  P = Spec[:,1:]/np.sum(Spec[:,1:], axis=1)[:, None]
  #entropy
  H = -np.sum(P*np.log2(P), axis=1)/np.log2((n//2))
  #centriod 
  C=np.sum(P*freqs[None, :], axis=1)
  #Absolute distance  of frequencies from from Centroid
  distC=np.abs((C[:,None]-freqs))
  #bandwidth is the weighted mean of the distance
  BW=np.sum(distC*P, axis=1)
  #maximum frequency 
  max_fr = freqs[np.argmax(Spec[:,1:], axis = 1)]
  df = np.concatenate([E[:,None], H[:,None], C[:,None],
                       BW[:,None], max_fr[:,None]], axis = 1)
  fun_names = ["_energy", "_entropy", "_centroid", "_bandwidth", "_max_freq"]                   
  return df, fun_names

#####################Training and evaluation functions##########################


def hierarchical_learn(classifier, meta_classes, X, y):
  """
  returns list of classifiers for hierarchical classification 
  [base classifiers, meta-class classifier 1, meta-class classifier ...]
  meta_class: list of meta-classes, each a list of target-classes
  X: training features
  y: target classes
  """
  clfs = []
  mult_classes = [classes for classes in meta_classes if len(classes) > 1]
  map_dict = {i:mc[0] for mc in meta_classes for i in mc}
  y_ = np.array([map_dict[i] for i in y])#converting y to meta-classes
  base = classifier()
  scaler = StandardScaler()
  X_base = scaler.fit_transform(X)
  base.fit(X_base, y_)
  clfs.append(base)
  for classes in mult_classes:#training each classifier discriminating within meta-class
    ind = np.isin(y, classes)
    clf = classifier()
    scaled_X = scaler.fit_transform(X[ind,:])
    clf.fit(scaled_X, y[ind])
    clfs.append(clf)
  return clfs

def hierarchical_predict(X, clfs):  
  """
  predicts y for X given a hierarchical system
   clfs: list of classifiers [base classifier, meta-class classifier 1,
    meta-class classifier 2...]
  meta-class: list of meta-classes each containing a list of target classes
  """
  scaler = StandardScaler()
  X_base = scaler.fit_transform(X)
  y_ = clfs[0].predict(X_base)#predicting meta classes (base)
  for clf in clfs[1:]:#discriminating within meta-classes
    ind = np.isin(y_, clf.classes_)
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X[ind,:])
    y_pred = clf.predict(scaled_X)
    y_[ind] = y_pred
  return y_