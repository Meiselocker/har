from scipy import signal as sg
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd


def get_AC_DC(X, sf, order = 1, crit_freq = 2):
  """
  returns high-pass and low-pass filtered signals using butterworth filter
  X: original signal
  sf = sampling frequency
  order: order of filter
  crit_freq: critical frequency of filter
  """
  
  colnames = X.columns
  AC_names = [colname + "_AC" for colname in colnames]
  DC_names = [colname + "_DC" for colname in colnames]
  X = X.values
  sos_low = sg.butter(order, crit_freq, 'lp', fs = sf, output = 'sos')
  sos_high = sg.butter(order, crit_freq, 'hp', fs = sf, output = 'sos')
  AC = np.zeros(X.shape)
  DC = np.zeros(X.shape)
  for col in range(X.shape[1]):
    AC [:,col] = sg.sosfilt(sos_low, X[:,col])
    DC [:,col] = sg.sosfilt(sos_high, X[:,col])
  return pd.DataFrame(AC, columns = AC_names), pd.DataFrame(DC, columns = DC_names)

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


def set_clf():
    """
    Returns a stacking classifier based on SVC, RF, and LR. 

    Returns
    -------
    clf : sklearn classifier object

    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import StackingClassifier
    
    svc = SVC(probability = True, class_weight="balanced")
    rf = RandomForestClassifier(class_weight="balanced", n_estimators = 500 ,
                                n_jobs=-1)
    lr = LogisticRegression(multi_class = "auto", penalty = 'l2', solver = "lbfgs", n_jobs = -1)
    clf = StackingClassifier(estimators = [('svc', svc), ('rf', rf), ('lr', lr)])
    return clf

