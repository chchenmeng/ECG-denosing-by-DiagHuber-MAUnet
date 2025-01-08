from torch.utils.data import DataLoader, Dataset
import torch
import pickle
import numpy as np
import torch.optim as optim
from Mamba_EncDec import RM_UNet, MAUnet
import torch
from tqdm import tqdm
import pywt
from ptflops import get_model_complexity_info
from torchsummary import summary
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
import scipy.fftpack as fftpack

# PyTorch Huber loss
huber_loss = torch.nn.HuberLoss(delta=0.05)
def hann_window(length):
    n = torch.arange(length, dtype=torch.float32).cuda()  # Move to GPU
    window = 0.5 - 0.5 * torch.cos((2.0 * np.pi * n) / (length - 1))
    return window


def rfftfreq(n, d=1.0):
    return torch.fft.rfftfreq(n, d=d).cuda()  # Move to GPU


def periodogram(signal, sample_rate, window='hann', nfft=None, scaling='density'):
    length = signal.shape[-1]
    window_func = hann_window(length)
    windowed_signal = signal * window_func

    # Compute the DFT
    dft = torch.fft.fft(windowed_signal, n=length, norm="backward")

    # Compute the power spectrum
    power_spectrum = torch.abs(dft) ** 2

    # Normalize the power spectrum
    if scaling == 'density':
        power_spectrum /= sample_rate
    elif scaling == 'spectrum':
        power_spectrum /= torch.sum(window_func) ** 2
    elif scaling == 'magnitude':
        power_spectrum = torch.sqrt(power_spectrum)

    # Compute frequencies
    frequencies = rfftfreq(length, 1 / sample_rate)
    return frequencies, power_spectrum


def combined_huber_freq_loss(y_true, y_pred, delta=0.05, sample_rate=360):
    y_true, y_pred = y_true.cuda(), y_pred.cuda()  # Move tensors to GPU

    # Compute the periodograms for true and predicted values
    _, power_spectrum_orig = periodogram(y_true, sample_rate)
    _, power_spectrum_denoised = periodogram(y_pred, sample_rate)

    # Compute similarity using cosine similarity
    similarity = F.cosine_similarity(power_spectrum_orig, power_spectrum_denoised, dim=-1).mean()
    frequency_weights = torch.exp(1 - torch.abs(similarity))

    # Calculate Huber loss with frequency weighting
    diff = y_true - y_pred
    squared_loss = diff ** 2
    linear_loss = delta * (torch.abs(diff) - 0.5 * delta)

    weighted_loss = frequency_weights * torch.where(torch.abs(diff) <= delta, squared_loss, linear_loss)

    # Calculate the average loss
    loss = weighted_loss.mean() - F.cosine_similarity(y_pred, y_true, dim=-1).mean()
    return loss


def combined_ssd_mad_loss(y_true, y_pred):
    mad_loss = torch.max((y_true - y_pred) ** 2, dim=-2).values * 50
    ssd_loss = torch.sum((y_true - y_pred) ** 2, dim=-2)
    return (mad_loss + ssd_loss).mean()


def prd_loss(y_true, y_pred):
 
    N = torch.sum((y_pred - y_true) ** 2, dim=-1)  ##dim -2

    D = torch.sum(y_true ** 2, dim=-1)

    prd = torch.sqrt(N / D)

    return prd.mean()

def DiagHuber(original, denoised, level=4):
    # Wavelet decomposition
    coeffs_original = pywt.wavedec(original.detach().cpu().numpy(), 'db1', level=level)
    coeffs_denoised = pywt.wavedec(denoised.detach().cpu().numpy(), 'db1', level=level)

    # # weights
    w_AL1 = level + 1
    w_Dj1 = torch.arange(1, level + 1, dtype=torch.float32).to(original.device)

    # Approximation subband
    approx_numerator = torch.tensor(np.sum((coeffs_original[0] - coeffs_denoised[0]) ** 2), dtype=torch.float32).to(
        original.device)
    approx_denominator = torch.tensor(np.sum(coeffs_original[0] ** 2), dtype=torch.float32).to(original.device)

    if approx_denominator == 0:
        approx_mse = torch.tensor(0.0).to(original.device)
    else:
        approx_mse = approx_numerator / approx_denominator


    approx_psnr = torch.sqrt(approx_mse)

    # Detail subband
    detail_numerator = torch.tensor(
        np.sum([np.sum((coeffs_original[i] - coeffs_denoised[i]) ** 2) for i in range(1, level + 1)]),
        dtype=torch.float32).to(original.device)
    detail_denominator = torch.tensor(np.sum([np.sum(coeffs_original[i] ** 2) for i in range(1, level + 1)]),
                                      dtype=torch.float32).to(original.device)

    if detail_denominator == 0:
        detail_mse = torch.tensor(0.0).to(original.device)
    else:
        detail_mse = detail_numerator / detail_denominator

    detail_psnr = torch.sqrt(detail_mse)

    # ##modified weights****start************
    energy_AL = approx_denominator / torch.tensor(len(coeffs_original[0])).to(original.device)
    detail_denominator1 = torch.tensor([np.sum(coeffs_original[i] ** 2) for i in range(1, level + 1)], dtype=torch.float32).to(original.device)
    energy_DL = detail_denominator1/torch.tensor([len(coeffs_original[i]) for i in range(1, level+1)]).to(original.device)
    # print(energy_DL)
    energy_all = torch.sum(energy_DL).to(original.device)+energy_AL
    w_AL = energy_AL/energy_all
    # w_AL = (w_AL1)*(-w_AL.to(original.device)*torch.log10(w_AL)).to(original.device)  ##entropy
    # w_AL = -torch.log10(w_AL).to(original.device)  ##self information
    w_AL = (w_AL1**2)*w_AL.to(original.device) ##last

    w_Dj = energy_DL.to(original.device)/energy_all
    # w_Dj = (-w_Dj*torch.log10(w_Dj)).to(original.device)
    # w_Dj = (w_Dj1)*(-w_Dj*torch.log10(w_Dj)).to(original.device)  ##level*entropy
    w_Dj = (w_Dj1**2)*w_Dj.to(original.device)   ##last
    ###*************end********************
    #  MSEWPRD
    msewprd = w_AL * approx_psnr + torch.sum(w_Dj * detail_psnr)

    # msewprd = approx_psnr + torch.sum(detail_psnr)   ##PRD

    # Huber Loss 
    huber = huber_loss(denoised, original)

    # combined 
    overall_loss = msewprd + huber  ##
    # print(msewprd/1000)
    # print(huber*10)
    return overall_loss


# datasets define
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# training hyperparameters
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=5):
    model = model.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model, input_size=(1, 512)) 
    model_flops, _ = get_model_complexity_info(model, (1, 512))
    print('model flops: ', model_flops)
    early_stopping = EarlyStopping(patience=patience) 

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # show loss
            loop.set_postfix(loss=loss.item())

        # print loss
        train_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        val_loss = validate_model(model, val_loader, criterion, device)

        # earlystop
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print(f"Training stopped early at epoch {epoch + 1}")
            break
    return model


def validate_model(model, val_loader, criterion, device):
    model.eval()  # evaluate
    val_loss = 0.0
    with torch.no_grad():  # 
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss  # judge earlystopping


def calculate_rmse(predicted, target):
    return torch.sqrt(torch.mean((predicted - target) ** 2))


def calculate_snr(predicted, target):
    signal_power = torch.sum(target ** 2)
    noise_power = torch.sum((predicted - target) ** 2)
    return 10 * torch.log10(signal_power / noise_power)


def test_model(model, test_loader, criterion, device):
    model.eval()  
    total_loss = 0.0
    total_rmse = 0.0
    total_snr = 0.0
    num_batches = len(test_loader)

    original_data = []  
    noisy_data = []  
    denoised_data = []  

    with torch.no_grad():  
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            
            rmse = calculate_rmse(outputs, targets)
            snr = calculate_snr(outputs, targets)

            total_rmse += rmse.item()
            total_snr += snr.item()

            # save data
            original_data.append(targets.cpu().numpy())
            noisy_data.append(inputs.cpu().numpy())
            denoised_data.append(outputs.cpu().numpy())


    avg_loss = total_loss / num_batches
    avg_rmse = total_rmse / num_batches
    avg_snr = total_snr / num_batches

    print(f"Test Loss (MSE): {avg_loss:.4f}")
    print(f"Test RMSE: {avg_rmse:.4f}")
    print(f"Test SNR: {avg_snr:.4f} dB")


    original_data = np.concatenate(original_data, axis=0)
    noisy_data = np.concatenate(noisy_data, axis=0)
    denoised_data = np.concatenate(denoised_data, axis=0)
    results = [noisy_data, original_data,  denoised_data]
    return results



# Create training and validation data loaders.

newpath = 'E:/PythonProject/denoiseProject/DeepFilter/data/'
#
newDataSavePath = 'E:/Learning/paper_loss/results/data/'  
newpath1 = 'E:/PythonProject/denoiseProject/DeepFilter/data/snr/'  ##test
dataLen = 512
noise_versions = [1,2] 


for nv in noise_versions:

    with open(newpath1 + '/snr_rmn_' + str(dataLen) + '_overlapping_dataset_nv' + str(nv) + '.pkl', 'rb') as input:
        Dataset = pickle.load(input)
    # ##delete DC component
    [X_train, y_train, X_test, y_test] = Dataset
    y_train = y_train - np.mean(y_train, axis=1)[:, np.newaxis]
    y_test = y_test - np.mean(y_test, axis=1)[:, np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
    # Dataset = [X_train, y_train, X_test, y_test]
    X_train = np.swapaxes(X_train, 1, 2)
    y_train = np.swapaxes(y_train, 1, 2)
    X_val = np.swapaxes(X_val, 1, 2)
    y_val = np.swapaxes(y_val, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    y_test = np.swapaxes(y_test, 1, 2)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # initialization
    # model = RM_UNet(dropout=0.1, activation="relu")
    model = MAUnet()

    # Define the loss function and optimizer.

    criterion = DiagHuber
    # criterion = torch.nn.MSELoss()
    # criterion = combined_ssd_mad_loss
    # criterion = combined_huber_freq_loss
    # criterion = prd_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model,  train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, num_epochs=10000, device="cuda" if torch.cuda.is_available() else "cpu")
    # test 
    test_results = test_model(
        model,
        test_loader=test_loader,
        criterion=criterion,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # # with open(newDataSavePath + 'wprdHuber_db1-4_512-RMUnet_norm_d64' + '_nv' + str(nv) + '.pkl','wb') as output:  # Overwrites any existing file.
    # #     pickle.dump(test_results, output)
    # #
    # with open(newDataSavePath + 'wprdHuber_db1-4_512-ACDAEM_d64_conv3_last_ssdmad2' + '_nv' + str(nv) + '.pkl','wb') as output:  # Overwrites any existing file.
    #     pickle.dump(test_results, output)

    # model.save('ACDAEM_mse_db1-4.h5')

    # #test icentia11k
    # # Test score 30min
    # import glob, scipy
    # import scipy.io as sio
    # from datetime import datetime
    # dataPath = 'E:/Denoise/icentia-ecg-master/saveTest_30min/'
    # filenames = glob.glob(dataPath + "*sig.mat")
    # for item in filenames:
    #     print(item)
    #     dataTest = sio.loadmat(item)['signal']
    #     start_test = datetime.now()
    #     X_test = np.zeros((1264, 512))
    #     dataTest1 = scipy.signal.resample_poly(dataTest[0, :], 360, 250)
    #     for i in range(1264):
    #         X_test[i, :] = dataTest1[i * 512:(i + 1) * 512]
    #     # print(X_test)
    #     X_test = np.expand_dims(X_test, axis=1)
    #     test_dataset_ = CustomDataset(X_test, X_test)
    #     test_loader_ = DataLoader(test_dataset_, batch_size=16, shuffle=False)
    #     denoised = []
    #     for inputs, targets in test_loader_:
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         # forward
    #         out = model(inputs)
    #         denoised.append(out.detach().cpu().numpy())
    #     denoised = np.array(denoised)
    #     y_restor = np.hstack((denoised.flatten(), dataTest1[-833:-1]))
    #     dataDenoised = scipy.signal.resample_poly(y_restor, 250, 360)
    #     sio.savemat('E:/Learning/paper_loss/results/Icentia11k/' + 'nv' + str(nv) + 'training2_/' + item[-12:-7] + 'filtered.mat',
    #                 {'filtered': dataDenoised})
    #     end_test = datetime.now()
    #     test_time_list = end_test - start_test
    # ##end


# ###test color noise
# import scipy
# import matplotlib.pyplot as plt
# dataPath = 'E:/Learning/paper2-multiTask/qualityAssess/geneTest/cinc2011Test/'
# savePath = 'XXXX'
# testBrnNoisy = np.load(dataPath + 'testBrnNoisy.npy')
# testBrnOrig = np.load(dataPath + 'testBrnOrig.npy')
# testPinkNoisy = np.load(dataPath + 'testPinkNoisy.npy')
# testPinkOrig = np.load(dataPath + 'testPinkOrig.npy')
# testBWNoisy = np.load(dataPath + 'testBWNoisy.npy')
# testBWOrig = np.load(dataPath + 'testBWOrig.npy')
# testWhiteNoisy = np.load(dataPath+'testWhiteNoisy.npy')
# testWhiteOrig = np.load(dataPath+'testWhiteOrig.npy')
# # bw_data = np.load(dataPath+'bw_data.npy')
# # data1 = data[0,:]/200   ##gain 200   ##105
# # data_test = data1[16*60*360:24*60*360]  #105
# denoiseSave = []  ##save 512 samples
# dataAllSave = []  ##all datalen
# saveDataAll = []
# for item in range(15): 
#     brnNoisy = testBrnNoisy[item, :]  ##noisy,Pink; Brn;BW
#     # brnNoisy = bw_data[item, :]   ##bw
#     brnNoisy = np.pad(brnNoisy, (0, 48), 'constant')
#     brnOrig = testBrnOrig[item, :]  ##orig
#     brnOrig = np.pad(brnOrig, (0, 48), 'constant')
#     data_res = scipy.signal.resample_poly(brnNoisy, 360, 200)
#     data_raw = scipy.signal.resample_poly(brnOrig, 360, 200)
#     ####
#     # data_test = normalized_data
#     # data_res = resample_poly(data_test,200,500)
#     # data_raw = resample_poly(brnOrig,200,500)
#     N = len(data_res) // 512
#     seg = 512
#     X_test = []
#     data_raw_seg = []
#     for k in range(N):
#         X_test.append(data_res[k * seg:(k + 1) * seg])
#         data_raw_seg.append(data_raw[k * seg:(k + 1) * seg])
#     X_test = np.expand_dims(np.array(X_test), axis=2)
#
#     X_test = np.expand_dims(X_test.squeeze(), axis=1)
#     test_dataset_ = CustomDataset(X_test, X_test)
#     test_loader_ = DataLoader(test_dataset_, batch_size=1, shuffle=False)
#     denoised = []
#     for inputs, targets in test_loader_:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         inputs, targets = inputs.to(device), targets.to(device)
#         # forward 
#         out = model(inputs)
#         denoised.append(out.detach().cpu().numpy())
#     denoised = np.array(denoised)
#     y_restor = np.concatenate((denoised.flatten(),data_raw[-104:-1]))
#
#     datasave = [np.transpose(X_test,(0,2,1)), np.expand_dims(np.array(data_raw_seg),axis=-1),np.expand_dims(denoised.squeeze(),axis=-1)]
#     saveDataAll.extend(datasave)
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.arange(len(data_res)) / 360, data_res, color='darkorange')
#     plt.plot(np.arange(len(data_res)) / 360, y_restor, color='limegreen')
#     plt.plot(np.arange(len(data_res)) / 360, data_raw, color='blue')
#     plt.legend(['ECG with -4dB brown noise', 'Denoised by MAUnet', 'Original ECG'], loc='upper right',fontsize=12)
#     plt.show(block=True)
#
#     plt.figure()
#
#     plt.psd(data_res, NFFT=128, Fs=360, color='darkorange')
#     plt.psd(y_restor, NFFT=128, Fs=360, color='limegreen')
#     plt.psd(data_raw, NFFT=128, Fs=360, color='blue')
#     plt.legend(['ECG with -4dB brown noise', 'Denoised by MAUnet', 'Original ECG'], loc='upper right',fontsize=12)
#     plt.tight_layout()
#     plt.show(block=True)
# reorganized_dalist = [[], [], []]
# for sublist in saveDataAll:
#     
#     for i in range(3):
#         reorganized_dalist[i].append(sublist[i])
#
# reorganized_dalist = [np.concatenate(sublist, axis=0) for sublist in reorganized_dalist]
# np.save(savePath + 'brown_mse.pkl', reorganized_dalist)
