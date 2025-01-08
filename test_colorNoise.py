import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle
import pandas as pd
import signalz
import signalz.generators.ecgsyn as ecgsyn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

def colored_noise(sd, length, fs, beta):
    #Colored noise generation by frequency domain filtering of white noise.
    #sd: the standard deviation of the generated noise
    #length: the noise vector length
    #fs: the sampling rate
    #beta: the noise color by assuming that the noise spectrum is propotional with 1/f^beta. beta = 0 corresponds to white noise.
  
    len_val = length + np.ceil(length / 10)
    len_val += 4 - len_val % 4
    halflen = np.ceil(len_val / 2)
    s = sd * np.random.randn(int(len_val))
    S = np.fft.fft(s, int(len_val))**2
    f = np.arange(0, len_val) * fs / len_val
    S[1:int(halflen) + 1] = S[1:int(halflen) + 1] / np.abs(f[1:int(halflen) + 1])**beta
    S[int(halflen) + 1:] = S[int(halflen) + 1:] / np.abs(f[int(halflen) - 1:0:-1])**beta
    n = np.real(np.fft.ifft(np.sqrt(S), int(len_val), axis=0))
    # A highpass filter for testing
    f1 = 0.1
    S[0] = 0
    k = np.arange(2, int(np.ceil(f1 * len_val / fs)))
    S[k] = 0
    S[int(len_val) - k + 1] = 0
    n = n[int(halflen - np.ceil(length / 2)):int(halflen + np.floor(length / 2))]
    n = sd * (n - np.mean(n)) / np.std(n)
    n = n.reshape(-1, 1)
    n = n.squeeze()
    return n



def string_to_number(s):
    split_parts = s.split(',')
    return int(split_parts[0])

def get_second_part(s):
    split_parts = s.split(',')
    return split_parts[1]

def add_noise_to_signal(signal, noise, snr):
    signal_power = np.mean(signal ** 2)

    noise_power = signal_power / (10 ** (snr / 10))
    adjusted_noise = np.sqrt(noise_power) * noise
    noisy_signal = signal + adjusted_noise

    return noisy_signal
cinc11path='E:/ECGDatabase/cinc2011/set-a/set-a/'
accept = pd.read_csv('E:/ECGDatabase/cinc2011/set-a/set-a_RECORDS-acceptable.txt', sep='\t', header=None)
accept = accept.to_numpy().squeeze()
accept = accept.tolist()
newFs = 360
seg = 512
# Preprocessing signals
namesPath = glob.glob(cinc11path + "/*.dat")

register_name = None
label = np.zeros(len(namesPath))
data = []
item = -1
for i in namesPath:
    item = item+1
    # reading signals
    aux = i.split('.dat')
    register_name = aux[0].split('/')[-1]
    name = int(register_name[-7:])
    if name in accept:
        label[item] = 0    ###0 is acceptable
    signal, fields = wfdb.rdsamp(aux[0])
    # auxSig = signal[:, 0]  

    ######**********######
    res = resample_poly(signal, newFs, fields['fs'])

    # plt.figure()
    # plt.plot(res)

    data.append(res)
data_single = np.array(data)[:,:,1]  ##lead II
##
# random 10 rows
clean_data = data_single[[15, 38, 41, 44, 53, 118, 127, 148, 154, 185, 196, 224, 538, 635, 662]]

bw_data = data_single[[76, 78, 82, 84, 179, 183, 192, 225, 262, 515, 565, 583, 613, 628, 650]]
snr = -8
testBrnNoisy = []
testBrnOrig = []
for item in clean_data:

    brownNoise = colored_noise(sd=1, length=2000, fs=360, beta=2.0)
    snr = snr+1
    brnData = add_noise_to_signal(item, brownNoise, snr)
    plt.figure()
    plt.plot(item,'b')
    plt.plot(brnData,'r')
    plt.legend(['orig','brown noisy with %.2f db'%snr])
    plt.show(block=True)
    testBrnOrig.append(item)
    testBrnNoisy.append(brnData)

snr = -8
testPinkNoisy = []
testPinkOrig = []
for item in clean_data:
    pink_noise = colored_noise(sd=1, length=2000, fs=200, beta=1.0)
    snr = snr + 1
    pinkData = add_noise_to_signal(item,pink_noise,snr)
    plt.figure()
    plt.plot(item, 'b')

    plt.plot(pinkData, 'darkorange')
    plt.legend(['orig','pink noisy with %.2f db'%snr])
    plt.show(block=True)
    testPinkOrig.append(item)
    testPinkNoisy.append(pinkData)


# #load NSTDB
with open('../data/Noise.pkl','rb') as input:
    nstdb = pickle.load(input)
##NSTDB
##bw noise
[bw_signals, _, _] = nstdb
bw_signals = np.array(bw_signals)
bw_noise_channel1_a = bw_signals[:, 0]
bw_noise_channel1 = resample_poly(bw_noise_channel1_a,200,360)
testBWNoisy = []
testBWOrig = []
snr = -8
for item in clean_data:
    bw_noise = bw_noise_channel1[(snr+8)*2000:(snr+9)*2000]
    snr = snr + 1
    BWData = add_noise_to_signal(item,bw_noise,snr)
    plt.figure()
    plt.plot(item, 'b')

    plt.plot(BWData, 'darkorange')
    plt.legend(['orig','BW noisy with %.2f db'%snr])
    plt.show(block=True)
    testBWOrig.append(item)
    testBWNoisy.append(BWData)

save = 'E:/Learning/paper-multiTask/qualityAssess/geneTest/cinc2011Test/'
testBrnNoisy = np.array(testBrnNoisy)
testBrnOrig = np.array(testBrnOrig)
testPinkNoisy = np.array(testPinkNoisy)
testPinkOrig = np.array(testPinkOrig)
testBWNoisy = np.array(testBWNoisy)
testBWOrig = np.array(testBWOrig)
# np.save(save+'testBrnNoisy.npy',testBrnNoisy)
# np.save(save+'testBrnOrig.npy',testBrnOrig)
# np.save(save+'testPinkNoisy.npy',testPinkNoisy)
# np.save(save+'testPinkOrig.npy',testPinkOrig)
np.save(save+'testBWNoisy.npy',testBWNoisy)
np.save(save+'testBWOrig.npy',testBWOrig)
# np.save(save+'bw_data.npy',bw_data)

