'''
This code was originally from https://github.com/schmiph2/pysepm
'''


import torch
import asteroid
import numpy as np
from torch.nn.functional import conv1d

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    print(result.shape)
    exit()                                            
    if window is not None:
        result = window * result
    return result
    
def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)

def segmental_snr_loss(enhanced, target, window_size=32, min_snr=-10, max_snr=35):
    
    assert not (len(enhanced.shape)!=2 or len(target.shape)!=2), 'tensors should have 2 dimensions'
    assert enhanced.shape==target.shape, 'enhanced and target tensor should have same shape'





    exit()
if __name__ == '__main__':
    device='cuda'
    batch_size=4
    length=16000*4
    enhanced=torch.randn((batch_size, length)).float().to(device)
    target=torch.randn((batch_size, length)).float().to(device)

    # segmental_snr=segmental_snr_loss(enhanced, target)

    segmental_snr=SNRseg(enhanced.cpu().numpy(), target.cpu().numpy(), 16000)
    print(segmental_snr)

