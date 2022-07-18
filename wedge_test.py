# %% Testing on Synthetic 2-D Wedge Models
# Function definitions modified from original code by Wes Hamlyn (https://github.com/seg/tutorials-2014/blob/master/1412_Tuning_and_AVO/tuning_prestack.py)
# Reference: Hamlyn, W. (2014). Thin beds, tuning, and AVO. The Leading Edge, 33(12), 1394-1396. https://doi.org/10.1190/tle33121394.1

## Convolutional model: y = Dx + noise
# dimensions of y and x same. m == n

import def_algorithms   # import BPI, FISTA, SBL-EM, NuSPAN-1, NuSPAN-2
import def_models       # import trace/wedge models
import def_figs         # import trace/wedge figures definitions

import numpy as np
np.random.seed(4)
import time
import scipy

import argparse
parser = argparse.ArgumentParser(description='Testing on Synthetic 2-D Wedge Models')
parser.add_argument('-b', '--bpi', help='use BPI', action='store_true')
parser.add_argument('-f', '--fista', help='use FISTA', action='store_true')
parser.add_argument('-s', '--sblem', help='use SBL-EM', action='store_true')
parser.add_argument('-n1', '--nuspan1', help='use NuSPAN-1', action='store_true')
parser.add_argument('-n2', '--nuspan2', help='use NuSPAN-2', action='store_true')
args = parser.parse_args()

# %% Model Parameters

wvlt_type = 'ricker'  # 'ricker' or 'bandpass'
wvlt_cfreq = 30       # wavelet central frequency in Hz
model_type = 'np'       # 'nn' 'pp' 'np' or 'pn' --> polarity of upper and lower interface (p=positive, n=negative)

## Generate the seismic traces (y), dictionary/kernel matrix (D), reflectivity function (x), and the time axis
y, D_trace, x, time_axis, wvlt_t, wvlt_amp, lyr_times, lyr_indx = def_models.wedge(wvlt_type, wvlt_cfreq,model_type)
y = y.T
x = x.T
[num_samples, num_traces] = x.shape
print('Data Generated')

## Generate noisy signal (y_noisy)
target_snr_db = 10
y_noisy_list = []
for i in range(num_traces):
    y_noisy_list.append(def_models.trace_noisy(y[:, i], target_snr_db=target_snr_db))
y_noisy = np.array(y_noisy_list).T
print('Noisy Data Generated')
print('-'*50)

#%% Input Parameters
time_start = time.time()
x_hat = []

if args.bpi:
    dataset_method = 'wedge_bpi'

    lambd = 2.e-1
    maxit = 8
    tol = 1.e-4

    print('lambda', lambd)
    print('iterations', maxit)
    print('tolerance', tol)

    for i in range(num_traces):
        x_hat.append(def_algorithms.bpi(D_trace, y_noisy[:,i], num_samples, maxit, lambd, tol))
        if (i+1) % 5 == 0:
            print('Trace {} done'.format(i+1))
    x_hat = np.squeeze( np.array(x_hat) ).T

if args.fista:
    dataset_method = 'wedge_fista'
    
    lambd = 3.e-1
    maxit = 400
    print('lambda', lambd)
    print('iterations', maxit)
    
    for i in range(num_traces):
        x_hat.append(def_algorithms.fista(D_trace, y_noisy[:,i], lambd, maxit))
        if (i+1) % 5 == 0:
            print('Trace {} done'.format(i+1))
    x_hat = np.array(x_hat).T

if args.sblem:
    dataset_method = 'wedge_sblem'

    flag1 = 2
    flag3 = 0
    lambd = 6.e-2
    maxit = 80

    print('lambda', lambd)
    print('iterations', maxit)

    x_hat = []
    for i in range(num_traces):
        x_hat.append(def_algorithms.sbl_em(D_trace, y_noisy[:,i], lambd, maxit, flag1, flag3))
        if (i+1) % 5 == 0:
            print('Trace {} done'.format(i+1))
    x_hat = np.squeeze( np.array(x_hat) ).T

if args.nuspan1:
    dataset_method = 'wedge_nuspan1'

    L = scipy.linalg.norm(D_trace, ord=2) ** 2
    m, n = num_samples, num_samples

    lambda1_int = 1.0
    mu1_int = 1.0
    gama1_int = 2
    nu1_int = 1.0
    a1_int = 3

    learning_rate = 1.e-3

    # maxit = 10
    maxit = 15
    import torch
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = def_algorithms.nuspan1(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit)
    model = model.float().to(device)

    from pathlib import Path
    ckp_path = Path('nuspan1_trace_wedge_{}.pt'.format(maxit))
    ## Optimizer, Criterion, and Scheduler
    from torch import optim
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    if ckp_path.is_file():
        model, opt, start_epoch = def_algorithms.load_ckp(ckp_path, model, opt)
        model.eval()
    else:
        print('Enter correct model details')

    y_noisy_t = torch.from_numpy(y_noisy.T)
    if len(y_noisy.shape) <= 1:
        y_noisy_t = t_test_t.view(1, -1)
    y_noisy_t = y_noisy_t.float().to(device)

    x_out = np.zeros_like(y_noisy.T)
    with torch.no_grad():
        x_out = model(y_noisy_t.float())
        if len(y_noisy.shape) <= 1:
            x_out = x_out.view(-1)
        x_hat = (x_out.cpu().numpy()).T

    ### Re-estimating the amplitudes over the supports given by NuSPAN-1 (comment out to see results without re-estimating the amplitudes)
    x_hat_pinv = []
    for i in range(num_traces):
        x_hat_nnz = np.nonzero( x_hat[:, i] )
        H = D_trace.copy()
        H = H[:, np.r_[x_hat_nnz]]
        H_inv = np.linalg.pinv(H)

        amplitudes = np.dot(H_inv, y_noisy[:,i])
        amplitudes = np.array(amplitudes).T

        x_hat_pinv_trace = x_hat[:,i].copy()
        x_hat_pinv_trace[np.r_[x_hat_nnz]] = amplitudes[:]
        if any(np.abs(x_hat_pinv_trace - x_hat[:, i]) >= 0.33):
            x_hat_pinv_trace = x_hat[:,i]
        x_hat_pinv.append(x_hat_pinv_trace)
    x_hat = np.array(x_hat_pinv).T
    ### (comment out upto here to see results without re-estimating the amplitudes)

if args.nuspan2:
    dataset_method = 'wedge_nuspan2'

    L = scipy.linalg.norm(D_trace, ord=2) ** 2
    m, n = num_samples, num_samples

    lambda1_int = 1.0
    mu1_int = 1.0
    gama1_int = 2
    nu1_int = 1.0
    a1_int = 3

    learning_rate = 1.e-3

    maxit = 10
    import torch
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = def_algorithms.nuspan2(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit)
    model = model.float().to(device)

    from pathlib import Path
    ckp_path = Path('nuspan2_trace_wedge.pt')
    ## Optimizer, Criterion, and Scheduler
    from torch import optim
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    if ckp_path.is_file():
        model, opt, start_epoch = def_algorithms.load_ckp(ckp_path, model, opt)
        model.eval()
    else:
        print('Enter correct model details')

    y_noisy_t = torch.from_numpy(y_noisy.T)
    if len(y_noisy.shape) <= 1:
        y_noisy_t = t_test_t.view(1, -1)
    y_noisy_t = y_noisy_t.float().to(device)

    x_out = np.zeros_like(y_noisy.T)
    with torch.no_grad():
        x_out = model(y_noisy_t.float())[0]
        if len(y_noisy.shape) <= 1:
            x_out = x_out.view(-1)
        x_hat = (x_out.cpu().numpy()).T
    
    ### Re-estimating the amplitudes over the supports given by NuSPAN-1 (comment out to see results without re-estimating the amplitudes)
    x_hat_pinv = []
    for i in range(num_traces):
        x_hat_nnz = np.nonzero( x_hat[:, i] )
        H = D_trace.copy()
        H = H[:, np.r_[x_hat_nnz]]
        H_inv = np.linalg.pinv(H)

        amplitudes = np.dot(H_inv, y_noisy[:,i])
        amplitudes = np.array(amplitudes).T

        x_hat_pinv_trace = x_hat[:,i].copy()
        x_hat_pinv_trace[np.r_[x_hat_nnz]] = amplitudes[:]
        if any(np.abs(x_hat_pinv_trace - x_hat[:, i]) >= 0.33):
            x_hat_pinv_trace = x_hat[:,i]
        x_hat_pinv.append(x_hat_pinv_trace)
    x_hat = np.array(x_hat_pinv).T
    ### (comment out upto here to see results without re-estimating the amplitudes)

time_end = np.round((time.time() - time_start), 4)

y_hat = []
for i in range(num_traces):
    y_hat.append(np.dot(D_trace, x_hat[:, i]))
y_hat = np.array(y_hat).T

cc = []
rre = []
srer = []
pes = []

for i in range(num_traces):
    cc.append(np.abs(np.corrcoef(x[:, i], x_hat[:, i])[0, 1]))

    error_num = scipy.linalg.norm(x[:, i]) ** 2
    error_den = scipy.linalg.norm(x_hat[:, i] - x[:, i]) ** 2
    rre.append(error_den / error_num)
    srer.append(10 * np.log10(error_num / error_den))
    
    pes.append(def_algorithms.pes(x[:, i], x_hat[:, i]))

print('-'*50)
cc = np.round( np.nanmean(np.array(cc)), 4 )
print('Correlation Coefficient (Reflectivity): {}'.format(cc))

rre = np.round( np.nanmean(np.array(rre)), 4 )
print('Relative Reconstruction Error: {}'.format(rre))

srer = np.round( np.nanmean(np.array(srer)), 4 )
print('Signal-to-Reconstruction Error Ratio: {} dB'.format(srer))

pes = np.round( np.nanmean(np.array(pes)), 4 )
print('Probability of Error in Support: {}'.format(pes))

print('Time: ', time_end)
print('-'*50)

if args.bpi or args.fista or args.sblem:
    all_params = [dataset_method, model_type, num_samples, num_traces, wvlt_cfreq, target_snr_db, lambd, maxit, cc, rre, srer, pes, time_end]
if args.nuspan1 or args.nuspan2:
    all_params = [dataset_method, model_type, num_samples, num_traces, wvlt_cfreq, target_snr_db, np.nan, maxit, cc, rre, srer, pes, time_end]

import csv
with open('test_wedge.csv', mode='a') as params:
    params_writer = csv.writer(params, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    params_writer.writerow(all_params)

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use("TkAgg")
font = {'size' : 4}
matplotlib.rc('font', **font)
csfont = {'fontname':'Times New Roman'}

# def_figs.figure_wedge(dataset_method, x.T, time_axis, model_type, 'Reflectivity True', num=1)
# def_figs.figure_wedge(dataset_method, y.T, time_axis, model_type, 'Seismic Traces True', num=2)
# def_figs.figure_wedge(dataset_method, y_noisy.T, time_axis, model_type, 'Seismic Traces Noisy', num=3)
def_figs.figure_wedge(dataset_method, x_hat.T, time_axis, model_type, 'Reflectivity Predicted', num=4)
# def_figs.figure_wedge(dataset_method, y_hat.T, time_axis, model_type, 'Seismic Traces Predicted', num=5)
# plt.close('all')

print('*'*10)
print('ALL DONE')
print('*'*10)
