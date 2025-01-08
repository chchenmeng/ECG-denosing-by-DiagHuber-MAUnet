import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.patches as mpatches
from metrics import MAD, SSD, PRD, COS_SIM, RMSE, SNR, WPRD
import pywt
import itertools

sigLen = 512

newDataSavePath = './comparation/'
newDataSavePath2= 'E:/Learning/paper_loss/results/data/'
dataLen = 512
dl_experiments = [
    'DRNN',
    'Multibranch LANLD',  ##DeepFilter
    'ACDAE',
    'CBAM-DAE',
    'Transformer_DAE',
    'MAUnet'
]

with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[0] + '_nv1.pkl',
          'rb') as input:
    test_DRNN_nv1 = pickle.load(input)
with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[0] + '_nv2.pkl',
          'rb') as input:
    test_DRNN_nv2 = pickle.load(input)
test_DRNN = [np.concatenate((test_DRNN_nv1[0], test_DRNN_nv2[0])),
             np.concatenate((test_DRNN_nv1[1], test_DRNN_nv2[1])),
             np.concatenate((test_DRNN_nv1[2], test_DRNN_nv2[2]))]


with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[2] + '_nv1.pkl',
          'rb') as input:
    test_ACDAE_nv1 = pickle.load(input)
with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[2] + '_nv2.pkl',
          'rb') as input:
    test_ACDAE_nv2 = pickle.load(input)

test_ACDAE = [np.concatenate((test_ACDAE_nv1[0], test_ACDAE_nv2[0])),
                np.concatenate((test_ACDAE_nv1[1], test_ACDAE_nv2[1])),
                np.concatenate((test_ACDAE_nv1[2], test_ACDAE_nv2[2]))]

with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[3] + '_nv1.pkl',
          'rb') as input:
    test_CBAM_DAE_nv1 = pickle.load(input)
with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[3] + '_nv2.pkl',
          'rb') as input:
    test_CBAM_DAE_nv2 = pickle.load(input)

test_CBAM_DAE = [np.concatenate((test_CBAM_DAE_nv1[0], test_CBAM_DAE_nv2[0])),
                np.concatenate((test_CBAM_DAE_nv1[1], test_CBAM_DAE_nv2[1])),
                np.concatenate((test_CBAM_DAE_nv1[2], test_CBAM_DAE_nv2[2]))]

with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[1] + '_nv1.pkl',
          'rb') as input:
    test_Multibranch_LANLD_nv1 = pickle.load(input)
with open(newDataSavePath + '/snr_rmn_test2_deleteDC_' + str(dataLen) + '-' + dl_experiments[1] + '_nv2.pkl',
          'rb') as input:
    test_Multibranch_LANLD_nv2 = pickle.load(input)

test_Multibranch_LANLD = [np.concatenate((test_Multibranch_LANLD_nv1[0], test_Multibranch_LANLD_nv2[0])),
                          np.concatenate((test_Multibranch_LANLD_nv1[1], test_Multibranch_LANLD_nv2[1])),
                          np.concatenate((test_Multibranch_LANLD_nv1[2], test_Multibranch_LANLD_nv2[2]))]
# Load Results Transformer_DAE
with open(newDataSavePath+'/snr_rmn_test2_deleteDC_'+str(dataLen) + '-' + dl_experiments[4] + 'CTransformer_nv1.pkl', 'rb') as input:
    test_Transformer_DAE_nv1 = pickle.load(input)
with open(newDataSavePath+'/snr_rmn_test2_deleteDC_'+str(dataLen) + '-' + dl_experiments[4] + 'CTransformer_nv2.pkl', 'rb') as input:
    test_Transformer_DAE_nv2 = pickle.load(input)
test_Transformer_DAE = [np.concatenate((test_Transformer_DAE_nv1[0], test_Transformer_DAE_nv2[0])),
                    np.concatenate((test_Transformer_DAE_nv1[1], test_Transformer_DAE_nv2[1])),
                    np.concatenate((test_Transformer_DAE_nv1[2], test_Transformer_DAE_nv2[2]))]



# Load Results MAUnet
with open(newDataSavePath2+'/mse_db1-4_512-MAUnet_d64_conv3_last_nv1.pkl', 'rb') as input:
    test_MAUnet_nv1 = pickle.load(input)
with open(newDataSavePath2+'/mse_db1-4_512-MAUnet_d64_conv3_last_nv2.pkl', 'rb') as input:
    test_MAUnet_nv2 = pickle.load(input)
test_MAUnet = [np.concatenate((test_MAUnet_nv1[0], test_MAUnet_nv2[0])),
                    np.concatenate((test_MAUnet_nv1[1], test_MAUnet_nv2[1])),
                    np.concatenate((test_MAUnet_nv1[2], test_MAUnet_nv2[2]))]


with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[0] + '_nv1.pkl',
          'rb') as input:
    test_DRNN_nv1 = pickle.load(input)
with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[0] + '_nv2.pkl',
          'rb') as input:
    test_DRNN_nv2 = pickle.load(input)
test_DRNN_new = [np.concatenate((test_DRNN_nv1[0], test_DRNN_nv2[0])),
             np.concatenate((test_DRNN_nv1[1], test_DRNN_nv2[1])),
             np.concatenate((test_DRNN_nv1[2], test_DRNN_nv2[2]))]


with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[2] + '_nv1.pkl',
          'rb') as input:
    test_ACDAE_nv1 = pickle.load(input)
with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[2] + '_nv2.pkl',
          'rb') as input:
    test_ACDAE_nv2 = pickle.load(input)

test_ACDAE_new = [np.concatenate((test_ACDAE_nv1[0], test_ACDAE_nv2[0])),
                np.concatenate((test_ACDAE_nv1[1], test_ACDAE_nv2[1])),
                np.concatenate((test_ACDAE_nv1[2], test_ACDAE_nv2[2]))]

with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[3] + '_nv1.pkl',
          'rb') as input:
    test_CBAM_DAE_nv1 = pickle.load(input)
with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[3] + '_nv2.pkl',
          'rb') as input:
    test_CBAM_DAE_nv2 = pickle.load(input)

test_CBAM_DAE_new = [np.concatenate((test_CBAM_DAE_nv1[0], test_CBAM_DAE_nv2[0])),
                np.concatenate((test_CBAM_DAE_nv1[1], test_CBAM_DAE_nv2[1])),
                np.concatenate((test_CBAM_DAE_nv1[2], test_CBAM_DAE_nv2[2]))]

with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[1] + '_nv1.pkl',
          'rb') as input:
    test_Multibranch_LANLD_nv1 = pickle.load(input)
with open(newDataSavePath2 + '/DiagHuber_db1-4_' + str(dataLen) + '-' + dl_experiments[1] + '_nv2.pkl',
          'rb') as input:
    test_Multibranch_LANLD_nv2 = pickle.load(input)

test_Multibranch_LANLD_new = [np.concatenate((test_Multibranch_LANLD_nv1[0], test_Multibranch_LANLD_nv2[0])),
                          np.concatenate((test_Multibranch_LANLD_nv1[1], test_Multibranch_LANLD_nv2[1])),
                          np.concatenate((test_Multibranch_LANLD_nv1[2], test_Multibranch_LANLD_nv2[2]))]
# Load Results Transformer_DAE
with open(newDataSavePath2+'/DiagHuber_db1-4_'+str(dataLen) + '-' + dl_experiments[4] + '_nv1.pkl', 'rb') as input:
    test_Transformer_DAE_nv1 = pickle.load(input)
with open(newDataSavePath2+'/DiagHuber_db1-4_'+str(dataLen) + '-' + dl_experiments[4] + '_nv2.pkl', 'rb') as input:
    test_Transformer_DAE_nv2 = pickle.load(input)
test_Transformer_DAE_new = [np.concatenate((test_Transformer_DAE_nv1[0], test_Transformer_DAE_nv2[0])),
                    np.concatenate((test_Transformer_DAE_nv1[1], test_Transformer_DAE_nv2[1])),
                    np.concatenate((test_Transformer_DAE_nv1[2], test_Transformer_DAE_nv2[2]))]
# Load Results MAUnet
with open(newDataSavePath2+'/DiagHuber_db1-4_'+str(dataLen) + '-' + 'MAUnet_d64_conv3_last_newweight_level2-relaltive-huber005_nv1.pkl', 'rb') as input:
    test_MAUnet_nv1 = pickle.load(input)
with open(newDataSavePath2+'/DiagHuber_db1-4_'+str(dataLen) + '-' + 'MAUnet_d64_conv3_last_newweight_level2-relaltive-huber005_nv2.pkl', 'rb') as input:
    test_MAUnet_nv2 = pickle.load(input)
test_MAUnet_new = [np.concatenate((test_MAUnet_nv1[0], test_MAUnet_nv2[0])),
                    np.concatenate((test_MAUnet_nv1[1], test_MAUnet_nv2[1])),
                    np.concatenate((test_MAUnet_nv1[2], test_MAUnet_nv2[2]))]



[X_test, y_test, y_pred] = test_Transformer_DAE
SSD_values_DL_Transformer_DAE = SSD(y_test, y_pred)
RMSE_values_DL_Transformer_DAE = RMSE(y_test, y_pred)
MAD_values_DL_Transformer_DAE = MAD(y_test, y_pred)
PRD_values_DL_Transformer_DAE = PRD(y_test, y_pred)
SNR_values_DL_Transformer_DAE = SNR(y_test, y_pred)
WPRD_values_DL_Transformer_DAE = np.zeros((y_test.shape[0], 1))
    
for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_Transformer_DAE[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_DRNN
SSD_values_DL_DRNN = SSD(y_test, y_pred)
RMSE_values_DL_DRNN = RMSE(y_test, y_pred)
MAD_values_DL_DRNN = MAD(y_test, y_pred)
PRD_values_DL_DRNN = PRD(y_test, y_pred)
SNR_values_DL_DRNN = SNR(y_test, y_pred)
WPRD_values_DL_DRNN = np.zeros((y_test.shape[0], 1))
   
for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_DRNN[i, 0] = WPRD(original, denoised, 5)

# Multibranch_LANLD
[X_test, y_test, y_pred] = test_Multibranch_LANLD
SSD_values_DL_exp_4 = SSD(y_test, y_pred)
RMSE_values_DL_exp_4 = RMSE(y_test, y_pred)
MAD_values_DL_exp_4 = MAD(y_test, y_pred)
PRD_values_DL_exp_4 = PRD(y_test, y_pred)
SNR_values_DL_exp_4 = SNR(y_test, y_pred)
WPRD_values_DL_exp_4 = np.zeros((y_test.shape[0], 1))
    
for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_exp_4[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_ACDAE

SSD_values_DL_ACDAE = SSD(y_test, y_pred)
RMSE_values_DL_ACDAE = RMSE(y_test, y_pred)
MAD_values_DL_ACDAE = MAD(y_test, y_pred)
PRD_values_DL_ACDAE = PRD(y_test, y_pred)
SNR_values_DL_ACDAE = SNR(y_test, y_pred)
WPRD_values_DL_ACDAE = np.zeros((y_test.shape[0], 1))
    
for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_ACDAE[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_CBAM_DAE

SSD_values_DL_CBAM_DAE = SSD(y_test, y_pred)
RMSE_values_DL_CBAM_DAE = RMSE(y_test, y_pred)
MAD_values_DL_CBAM_DAE = MAD(y_test, y_pred)
PRD_values_DL_CBAM_DAE = PRD(y_test, y_pred)
SNR_values_CBAM_DAE = SNR(y_test, y_pred)
WPRD_values_DL_CBAM_DAE = np.zeros((y_test.shape[0], 1))
   
for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_CBAM_DAE[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_MAUnet
y_test = y_test.transpose([0,2,1])
y_pred = y_pred.transpose([0,2,1])
SSD_values_DL_MAUnet = SSD(y_test, y_pred)
RMSE_values_DL_MAUnet = RMSE(y_test, y_pred)
MAD_values_DL_MAUnet = MAD(y_test, y_pred)
PRD_values_DL_MAUnet = PRD(y_test, y_pred)
SNR_values_MAUnet = SNR(y_test, y_pred)
WPRD_values_DL_MAUnet = np.zeros((y_test.shape[0], 1))
    
for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_MAUnet[i, 0] = WPRD(original, denoised, 5)
##*******************diaghuber**********************************
[X_test, y_test, y_pred] = test_Transformer_DAE_new
SSD_values_DL_Transformer_DAE_new = SSD(y_test, y_pred)
RMSE_values_DL_Transformer_DAE_new = RMSE(y_test, y_pred)
MAD_values_DL_Transformer_DAE_new = MAD(y_test, y_pred)
PRD_values_DL_Transformer_DAE_new = PRD(y_test, y_pred)
SNR_values_DL_Transformer_DAE_new = SNR(y_test, y_pred)
WPRD_values_DL_Transformer_DAE_new = np.zeros((y_test.shape[0], 1))

for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_Transformer_DAE_new[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_DRNN_new
SSD_values_DL_DRNN_new = SSD(y_test, y_pred)
RMSE_values_DL_DRNN_new = RMSE(y_test, y_pred)
MAD_values_DL_DRNN_new = MAD(y_test, y_pred)
PRD_values_DL_DRNN_new = PRD(y_test, y_pred)
SNR_values_DL_DRNN_new = SNR(y_test, y_pred)
WPRD_values_DL_DRNN_new = np.zeros((y_test.shape[0], 1))

for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_DRNN_new[i, 0] = WPRD(original, denoised, 5)

# Multibranch_LANLD
[X_test, y_test, y_pred] = test_Multibranch_LANLD_new
SSD_values_DL_exp_4_new = SSD(y_test, y_pred)
RMSE_values_DL_exp_4_new = RMSE(y_test, y_pred)
MAD_values_DL_exp_4_new = MAD(y_test, y_pred)
PRD_values_DL_exp_4_new = PRD(y_test, y_pred)
SNR_values_DL_exp_4_new = SNR(y_test, y_pred)
WPRD_values_DL_exp_4_new = np.zeros((y_test.shape[0], 1))

for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_exp_4_new[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_ACDAE_new
SSD_values_DL_ACDAE_new = SSD(y_test, y_pred)
RMSE_values_DL_ACDAE_new = RMSE(y_test, y_pred)
MAD_values_DL_ACDAE_new = MAD(y_test, y_pred)
PRD_values_DL_ACDAE_new = PRD(y_test, y_pred)
SNR_values_DL_ACDAE_new = SNR(y_test, y_pred)
WPRD_values_DL_ACDAE_new = np.zeros((y_test.shape[0], 1))

for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0] 
    WPRD_values_DL_ACDAE_new[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_CBAM_DAE_new
SSD_values_DL_CBAM_DAE_new = SSD(y_test, y_pred)
RMSE_values_DL_CBAM_DAE_new = RMSE(y_test, y_pred)
MAD_values_DL_CBAM_DAE_new = MAD(y_test, y_pred)
PRD_values_DL_CBAM_DAE_new = PRD(y_test, y_pred)
SNR_values_CBAM_DAE_new = SNR(y_test, y_pred)
WPRD_values_DL_CBAM_DAE_new = np.zeros((y_test.shape[0], 1))

for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0]  
    WPRD_values_DL_CBAM_DAE_new[i, 0] = WPRD(original, denoised, 5)

[X_test, y_test, y_pred] = test_MAUnet_new
y_test = y_test.transpose([0,2,1])
y_pred = y_pred.transpose([0,2,1])
SSD_values_DL_MAUnet_new = SSD(y_test, y_pred)
RMSE_values_DL_MAUnet_new = RMSE(y_test, y_pred)
MAD_values_DL_MAUnet_new = MAD(y_test, y_pred)
PRD_values_DL_MAUnet_new = PRD(y_test, y_pred)
SNR_values_MAUnet_new = SNR(y_test, y_pred)
WPRD_values_DL_MAUnet_new = np.zeros((y_test.shape[0], 1))

for i in range(y_test.shape[0]):
    original = y_test[i, :, 0]
    denoised = y_pred[i, :, 0] 
    WPRD_values_DL_MAUnet_new[i, 0] = WPRD(original, denoised, 5)

SSD_all = [
    SSD_values_DL_DRNN,
    SSD_values_DL_DRNN_new,
    SSD_values_DL_exp_4,   #DeepFilter
    SSD_values_DL_exp_4_new,
    SSD_values_DL_ACDAE,
    SSD_values_DL_ACDAE_new,
    SSD_values_DL_CBAM_DAE,
    SSD_values_DL_CBAM_DAE_new,
    SSD_values_DL_Transformer_DAE,
    SSD_values_DL_Transformer_DAE_new,
    SSD_values_DL_MAUnet,
    SSD_values_DL_MAUnet_new
]
RMSE_all = [
    RMSE_values_DL_DRNN,
    RMSE_values_DL_DRNN_new,
    RMSE_values_DL_exp_4,
    RMSE_values_DL_exp_4_new,
    RMSE_values_DL_ACDAE,
    RMSE_values_DL_ACDAE_new,
    RMSE_values_DL_CBAM_DAE,
    RMSE_values_DL_CBAM_DAE_new,
    RMSE_values_DL_Transformer_DAE,
    RMSE_values_DL_Transformer_DAE_new,
    RMSE_values_DL_MAUnet,
    RMSE_values_DL_MAUnet_new
]
MAD_all = [
    MAD_values_DL_DRNN,
    MAD_values_DL_DRNN_new,
    MAD_values_DL_exp_4,
    MAD_values_DL_exp_4_new,
    MAD_values_DL_ACDAE,
    MAD_values_DL_ACDAE_new,
    MAD_values_DL_CBAM_DAE,
    MAD_values_DL_CBAM_DAE_new,
    MAD_values_DL_Transformer_DAE,
    MAD_values_DL_Transformer_DAE_new,
    MAD_values_DL_MAUnet,
    MAD_values_DL_MAUnet_new
]

SNR_all = [
    SNR_values_DL_DRNN,
    SNR_values_DL_DRNN_new,
    SNR_values_DL_exp_4,
    SNR_values_DL_exp_4_new,
    SNR_values_DL_ACDAE,
    SNR_values_DL_ACDAE_new,
    SNR_values_CBAM_DAE,
    SNR_values_CBAM_DAE_new,
    SNR_values_DL_Transformer_DAE,
    SNR_values_DL_Transformer_DAE_new,
    SNR_values_MAUnet,
    SNR_values_MAUnet_new
]
WPRD_all = [
    WPRD_values_DL_DRNN,
    WPRD_values_DL_DRNN_new,
    WPRD_values_DL_exp_4,
    WPRD_values_DL_exp_4_new,
    WPRD_values_DL_ACDAE,
    WPRD_values_DL_ACDAE_new,
    WPRD_values_DL_CBAM_DAE,
    WPRD_values_DL_CBAM_DAE_new,
    WPRD_values_DL_Transformer_DAE,
    WPRD_values_DL_Transformer_DAE_new,
    WPRD_values_DL_MAUnet,
    WPRD_values_DL_MAUnet_new
]


################################################################################################################
# Segmentation by noise amplitude
newpath = 'E:/PythonProject/denoiseProject/DeepFilter/data/snr/'
rnd_test_nv1 = np.load(newpath + '/snr_rnd' + '/snr_rnd_test_' + str(dataLen) + '_nv1.npy')  
rnd_test_nv2 = np.load(newpath + '/snr_rnd' + '/snr_rnd_test_' + str(dataLen) + '_nv2.npy')

rnd_test = np.concatenate([rnd_test_nv1, rnd_test_nv1])

segm = [-6, 0, 6, 12, 18]
zero = []
SSD_seg_all = []
RMSE_seg_all = []
MAD_seg_all = []
WPRD_seg_all = []
SNR_seg_all = []
exp_name = [
    'DRNN',
    'DRNN_new',
    'DeepFilter',
    'DeepFilter_new',
    'ACDAE',
    'ACDAE_new',
    'CBAM-DAE',
    'CBAM-DAE_new',
    'TCDAE',
    'TCDAE_new',
    'MAUnet',
    'MAUnet_new'
]
for idx_exp in range(len(exp_name)):
    SSD_seg = [None] * (len(segm) - 1)
    RMSE_seg = [None] * (len(segm) - 1)
    MAD_seg = [None] * (len(segm) - 1)
    WPRD_seg = [None] * (len(segm) - 1)
    WDD_seg = [None] * (len(segm) - 1)
    SNR_seg = [None] * (len(segm) - 1)
    for idx_seg in range(len(segm) - 1):
        SSD_seg[idx_seg] = []
        RMSE_seg[idx_seg] = []
        MAD_seg[idx_seg] = []
        WPRD_seg[idx_seg] = []
        SNR_seg[idx_seg] = []
        # WDD_seg[idx_seg] = []
        for idx in range(len(rnd_test)):
            if idx not in zero:
                # Object under analysis (oua)
                # SSD
                oua = SSD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SSD_seg[idx_seg].append(oua)
                # SSD
                oua = RMSE_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    RMSE_seg[idx_seg].append(oua)
                # MAD
                oua = MAD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAD_seg[idx_seg].append(oua)

                # WPRD
                oua = WPRD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    WPRD_seg[idx_seg].append(oua)


                # SNR
                oua = SNR_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SNR_seg[idx_seg].append(oua)


    # Processing the last index
    # SSD
    SSD_seg[-1] = []
    for idx in range(len(rnd_test)):
        # Object under analysis
        oua = SSD_all[idx_exp][idx]
        if rnd_test[idx] > segm[-2]:
            SSD_seg[-1].append(oua)

    SSD_seg_all.append(SSD_seg)  # [exp][seg][item]
    # SSD
    RMSE_seg[-1] = []
    for idx in range(len(rnd_test)):
        # Object under analysis
        oua = RMSE_all[idx_exp][idx]
        if rnd_test[idx] > segm[-2]:
            RMSE_seg[-1].append(oua)

    RMSE_seg_all.append(RMSE_seg)  # [exp][seg][item]
    # MAD
    MAD_seg[-1] = []
    for idx in range(len(rnd_test)):
        # Object under analysis
        oua = MAD_all[idx_exp][idx]
        if rnd_test[idx] > segm[-2]:
            MAD_seg[-1].append(oua)

    MAD_seg_all.append(MAD_seg)  # [exp][seg][item]

    # PRD
    WPRD_seg[-1] = []
    for idx in range(len(rnd_test)):
        # Object under analysis
        oua = WPRD_all[idx_exp][idx]
        if rnd_test[idx] > segm[-2]:
            WPRD_seg[-1].append(oua)

    WPRD_seg_all.append(WPRD_seg)  # [exp][seg][item]


    # SNR
    SNR_seg[-1] = []
    for idx in range(len(rnd_test)):
        # Object under analysis
        oua = SNR_all[idx_exp][idx]
        if rnd_test[idx] > segm[-2]:
            SNR_seg[-1].append(oua)

    SNR_seg_all.append(SNR_seg)  # [exp][seg][item]

# Printing Tables
seg_table_column_name = []
for idx_seg in range(len(segm) - 1):
    column_name = str(segm[idx_seg]) + ' < noise < ' + str(segm[idx_seg + 1])
    seg_table_column_name.append(column_name)


# Step 1: Flatten the single-element lists within each sublist
flattened_data = []
for method in SNR_seg_all:
    method_partitions = []
    for partition in method:
        # Flatten each partition from a list of single-element lists to a single list of values
        flattened_partition = list(itertools.chain.from_iterable(partition))
        method_partitions.append(flattened_partition)
    flattened_data.append(method_partitions)
# Step 2: Organize data by partition across all methods for the box plot
data_by_partition = [
    [flattened_data[method][partition] for method in range(12)]
    for partition in range(4)
]
# Define macaron colors and hatch patterns for each method pair
macaron_colors = ['#FFB6C1', '#FFDAB9', '#FFFACD', '#C1FFC1', '#ADD8E6', '#E6E6FA']  # Colors for pairs
patterns = ['', '//']  # Patterns for original and improved versions
# Create custom legend patches
legend_patches = []
label = ['DRDNN','DeepFilter','ACDAE','CBAM-DAE','TCDAE','MAUnet']
for i in range(6):  # Five pairs
    color_patch = mpatches.Patch(color=macaron_colors[i], label=label[i])
    legend_patches.append(color_patch)
for pattern in patterns:
    style_patch = mpatches.Patch(facecolor="gray", hatch=pattern, label='Original' if pattern == '' else 'DiagHuber')
    legend_patches.append(style_patch)
# Step 3: Create individual figures for each partition
for i, partition_data in enumerate(data_by_partition):
    fig, ax = plt.subplots(figsize=(6, 3))
    # Plot boxplots with positions arranged to keep method pairs close
    positions = [j + offset for j in range(6) for offset in [0, 0.5]]
    bplot = ax.boxplot(
        [partition_data[method] for method in range(12)],
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.4
    )
    # Apply colors and hatching
    for j, patch in enumerate(bplot['boxes']):
        color_idx = j // 2  # Determine color pair
        patch.set_facecolor(macaron_colors[color_idx])
        patch.set_hatch(patterns[j % 2])  # Apply pattern for differentiation within pair
    # Title and labels
    # ax.set_title(f'Partition {i + 1} Results')
    ax.set_xticks([j + 0.25 for j in range(6)])  # Position x-ticks between pairs
    ax.set_xticklabels(['DRDNN','DeepFilter','ACDAE','CBAM-DAE','TCDAE','MAUnet'], rotation=0)
    ax.set_ylabel('SNR (dB)')
    # ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1, 1))
    # Add grid lines
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show(block=True)

