import numpy as np
import scipy.signal as sig
import pywt
from torch.utils.data import Dataset, DataLoader
import torch

'''
The customized dataset and data augmentations are defined here.
'''
class EEG_Dataset(Dataset):
    '''
    use use fold_idx to instantiate different train val datasets for k-fold cross validation
    '''
    def __init__ (self, X_train=None, y_train=None, p_train=None, X_val=None, y_val=None, p_val=None, X_test=None, y_test=None, p_test=None, mode='train'):
        if mode == 'train':
            self.X = X_train
            self.y = y_train- 769
            self.p = p_train
            
        elif mode == 'val':
            self.X = X_val
            self.y = y_val- 769
            self.p = p_val

        elif mode == 'test':
            self.X = X_test
            self.y = y_test - 769        
            self.p = p_test

    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, idx):
        '''
        X: (augmented) time sequence 
        y: class label
        p: person id
        '''
        X = torch.from_numpy(self.X[idx,:,:]).float()
        y = torch.tensor(self.y[idx]).long()
        p = torch.tensor(self.p[idx]).long()
        #p = torch.from_numpy(self.p[idx,:]).long()     
        sample = {'X': X, 'y': y, 'p':p}

        return sample


'''
Data Augmentions
'''

def standardize(X, mu, std):
    return (X - mu / std)


# Perform a Ms. Butterworth bandpass filter with the low and high frequencies specified
def bandpass_filter(X,low,high):
  N, C, T = X.shape
  out = np.zeros_like(X)
  nyq = 125 #nyquist frequency, highest able to be sensed for this data
  if high > nyq :
    high = nyq
  order = 9

  b, a = sig.butter(order, [low/nyq,high/nyq], btype='band')
  out = sig.lfilter(b, a, X, axis=-1)

  return out

def window_data(X, y, p, window_size, stride):
  '''
  X (a 3-d tensor) of size (#trials, #electrodes, #time series)
  y (#trials,): label 
  p (#trials, 1): person id
  X_new1: The first output stacks the windowed data in a new dimension, resulting 
    in a 4-d tensor of size (#trials x #electrodes x #windows x #window_size).
  X_new2: The second option makes the windows into new trails, resulting in a new
    X tensor of size (#trials*#windows x #electrodes x #window_size). To account 
    for the larger number of trials, we also need to augment the y data.
  y_new: The augmented y vector of size (#trials*#windows) to match X_new2.
  p_new: The augmented p vector of size (#trials*#windows) to match X_new2
 
  '''
  num_sub_trials = int((X.shape[2]-window_size)/stride)
  X_new1 = np.empty([X.shape[0],X.shape[1],num_sub_trials,window_size])
  X_new2 = np.empty([X.shape[0]*num_sub_trials,X.shape[1],window_size])
  y_new = np.empty([X.shape[0]*num_sub_trials])
  p_new = np.empty([X.shape[0]*num_sub_trials])
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      for k in range(num_sub_trials):
        X_new1[i,j,k:k+window_size]    = X[i,j,k*stride:k*stride+window_size]
        X_new2[i*num_sub_trials+k,j,:] = X[i,j,k*stride:k*stride+window_size]
        y_new[i*num_sub_trials+k] = y[i]
        p_new[i*num_sub_trials+k] = p[i]
  return X_new1, X_new2, y_new, p_new


def stft_data(X, window, stride):
    '''
    Inputs:
    X - input data, last dimension is one which transform will be taken across.
    window - size of sliding window to take transform across
    stride - stride of sliding window across time-series
    Returns:
    X_STFT - Output data, same shape as input with last dimension replaced with two new dimensions, F x T.
            where F = window//2 + 1 is the frequency axis
            and T = (input_length - window)//stride + 1, similar to the formula for aconvolutional filter.
    t - the corresponding times for the time axis, T
    f - the corresponding frequencies on the frequency axis, F.
    reshape X_STFT (N, C, F, T) to (N, C*F, T) to fit the input of rnn
    Note that a smaller window means only higher frequencies may be found, but give finer time resolution.
    Conversely, a large window gives better frequency resolution, but poor time resolution.
    '''
    noverlap = window-stride
    if noverlap < 0 :
        print('Stride results in skipped data!')
        return
    f, t, X_STFT = sig.spectrogram(X,nperseg=window,noverlap=noverlap,fs=250, return_onesided=True)
    N, C, F, T = X_STFT.shape
    X_STFT = X_STFT.reshape(N, C*F, T)
    return X_STFT


def cwt_data(X, num_levels, top_scale=3):
    '''
    Takes in data, computes CWT using the mexican hat or ricker wavelet using scipy
    Also takes in the top scale parameter.  I use logspace, so scale goes from 1 -> 2^top_scale with num_levels steps.
    Appends to the data a new dimension, of size 'num_levels'
    New dimension corresponds to wavelet content at num_levels different scalings (linear)
    also returns the central frequencies that the scalings correspond to
    input data is N x C X T
    output data is N x C x T x F
    note: CWT is fairly slow to compute
    # EXAMPLE USAGE
    test, freqs = cwt_data(X_train_valid[0:5,:,:],num_levels=75,top_scale=4)
    '''
    scales = np.logspace(start=0,stop=top_scale,num=num_levels)
    out = np.empty((X.shape[0],X.shape[1],X.shape[2],num_levels))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            coef = sig.cwt(X[i,j,:],sig.ricker,scales)
            out[i,j,:] = coef.T
    freqs = pywt.scale2frequency('mexh',scales)*250
    N, C, T, F = out.shape
    X_CWT = np.transpose(out, (0,1,3,2)).reshape(N, C*F, T)
    return X_CWT

def cwt_data2(X, y, p, num_levels, top_scale=3):
    '''
    Takes in data, computes CWT using the mexican hat or ricker wavelet using scipy
    Also takes in the top scale parameter.  I use logspace, so scale goes from 1 -> 2^top_scale with num_levels steps.
    Appends to the data a new dimension, of size 'num_levels'
    New dimension corresponds to wavelet content at num_levels different scalings (linear)
    also returns the central frequencies that the scalings correspond to
    input data is N x C X T
    output data is N x C x T x F
    note: CWT is fairly slow to compute
    # EXAMPLE USAGE
    test, freqs = cwt_data(X_train_valid[0:5,:,:],num_levels=75,top_scale=4)
    '''
    scales = np.logspace(start=0,stop=top_scale,num=num_levels)
    out = np.empty((X.shape[0],X.shape[1],X.shape[2],num_levels))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out[i,j,:] = sig.cwt(X[i,j,:],sig.ricker,scales).T
    #freqs = pywt.scale2frequency('mexh',scales)*250
    N, C, T, F = out.shape
    X_cwt = np.transpose(out, (0,3,1,2)).reshape(N*F, C, T)
    y_cwt = np.empty([X.shape[0]*F])
    p_cwt = np.empty([X.shape[0]*F])
    for i in range(X.shape[0]):
      for k in range(F):
        y_cwt[i*F+k] = y[i]
        p_cwt[i*F+k] = p[i]
    return X_cwt, y_cwt, p_cwt

def dwt_data(X,wav_name):
    wav = pywt.Wavelet(wav_name)
    n = X.shape[-1] #extract length of timeseries
    max_level = pywt.dwt_max_level(n,wav)
    out1 = np.concatenate(pywt.wavedec(X[0,:,:],wav,level=max_level,mode='zero'),axis=1)
    out = np.empty((X.shape[0],X.shape[1],out1.shape[1]))
    out[0,:,:] = out1
    for i in range(1,X.shape[0]):
        out[i,:,:] = np.concatenate(pywt.wavedec(X[i,:,:],wav,level=max_level,mode='zero'),axis=1)
    return out

def Aug_Data(X, y, p, aug_type=None, window_size=200, window_stride=20, stft_size=None, stft_stride=None, cwt_level=None, cwt_scale=None):
    if aug_type == None:
        X_aug, y_aug, p_aug = X, y, p
    elif aug_type == "window":
        _, X_aug, y_aug, p_aug = window_data(X, y, p, window_size, window_stride)
    elif aug_type == "stft":
        X_aug = stft_data(X, stft_size, stft_stride)
        y_aug, p_aug = y, p
    elif aug_type == "cwt":
        X_aug = cwt_data(X, cwt_level, cwt_scale)
        y_aug, p_aug = y, p
    elif aug_type == "cwt2":
        X_aug,y_aug,p_aug = cwt_data2(X, y, p, cwt_level, cwt_scale)
    elif aug_type == "dwt":
        X_aug = dwt_data(X,'sym9')
        y_aug, p_aug = y, p
    return X_aug, y_aug, p_aug
'''
Split training and validation
'''

def Train_Val_Data(X_train_valid, y_train_val):
    '''
    split the train_valid into k folds (we fix k = 5 here)
    return: list of index of train data and val data of k folds
    train_fold[i], val_fold[i] is the index for training and validation in the i-th fold 
    '''
    fold_idx = []
    train_fold = []
    val_fold = []
    train_val_num = X_train_valid.shape[0]
    fold_num = int(train_val_num / 5)
    perm = np.random.permutation(train_val_num)
    for k in range(5):
        fold_idx.append(np.arange(k*fold_num, (k+1)*fold_num, 1))
    for k in range(5):
        val_fold.append(fold_idx[k])
        count = 0
        for i in range(5):
            if i != k:
                if count == 0:
                    train_idx = fold_idx[i]
                else:
                    train_idx = np.concatenate((train_idx, fold_idx[i]))
                count += 1
        train_fold.append(train_idx)

    return train_fold, val_fold

