# %% Testing on Synthetic 1-D Seismic Traces

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
parser = argparse.ArgumentParser(description='Testing on Synthetic 1-D Seismic Traces')
parser.add_argument('-b', '--bpi', help='use BPI', action='store_true')
parser.add_argument('-f', '--fista', help='use FISTA', action='store_true')
parser.add_argument('-s', '--sblem', help='use SBL-EM', action='store_true')
parser.add_argument('-n1', '--nuspan1', help='use NuSPAN-1', action='store_true')
parser.add_argument('-n2', '--nuspan2', help='use NuSPAN-2', action='store_true')
args = parser.parse_args()

# %% Model Parameters

wvlt_type = 'ricker'  # 'ricker' or 'bandpass'
wvlt_cfreq = 30       # wavelet central frequency in Hz
num_samples = 300     # number of samples in a trace (must be > 128, length of the Ricker wavelet)
num_traces = 1000     # number of seismic traces
k = 0.05              # sparsity factor

## Generate the seismic trace (y), dictionary/kernel matrix (D), reflectivity function (x), and the time axis
y, D_trace, x, time_axis, wvlt_t, wvlt_amp = def_models.trace(wvlt_type, wvlt_cfreq, num_samples, num_traces, k)
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
print('SNR', target_snr_db)
print('Sparsity factor (k)', k)

time_start = time.time()
x_hat = []

if args.bpi:
    dataset_method = 'trace_bpi'
    name = 'BPI'

    lambd = 2.e-1
    maxit = 8
    tol = 1.e-4

    print('lambda', lambd)
    print('iterations', maxit)
    print('tolerance', tol)

    for i in range(num_traces):
        x_hat.append(def_algorithms.bpi(D_trace, y_noisy[:,i], num_samples, maxit, lambd, tol))
        if (i+1) % 100 == 0:
            print('Trace {} done'.format(i+1))
    x_hat = np.squeeze( np.array(x_hat) ).T

if args.fista:
    dataset_method = 'trace_fista'
    name = 'FISTA'
    
    lambd = 3.e-1
    maxit = 400
    print('lambda', lambd)
    print('iterations', maxit)
    
    for i in range(num_traces):
        x_hat.append(def_algorithms.fista(D_trace, y_noisy[:,i], lambd, maxit))
        if (i+1) % 100 == 0:
            print('Trace {} done'.format(i+1))
    x_hat = np.array(x_hat).T

if args.sblem:
    dataset_method = 'trace_sblem'
    name = 'SBL-EM'

    flag1 = 2
    flag3 = 0
    lambd = 6.e-2
    maxit = 80

    print('lambda', lambd)
    print('iterations', maxit)

    x_hat = []
    for i in range(num_traces):
        x_hat.append(def_algorithms.sbl_em(D_trace, y_noisy[:,i], lambd, maxit, flag1, flag3))
        if (i+1) % 100 == 0:
            print('Trace {} done'.format(i+1))
    x_hat = np.squeeze( np.array(x_hat) ).T

if args.nuspan1:
    dataset_method = 'trace_nuspan1'
    name = 'NuSPAN-1'

    L = scipy.linalg.norm(D_trace, ord=2) ** 2
    m, n = num_samples, num_samples

    lambda1_int = 1.0
    mu1_int = 1.0
    gama1_int = 2
    nu1_int = 1.0
    a1_int = 3

    learning_rate = 1.e-3

    maxit = 10
    # maxit = 15
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

if args.nuspan2:
    dataset_method = 'trace_nuspan2'
    name = 'NuSPAN-2'

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
    all_params = [dataset_method, num_samples, num_traces, wvlt_cfreq, k, target_snr_db, lambd, maxit, cc, rre, srer, pes, time_end]
if args.nuspan1 or args.nuspan2:
    all_params = [dataset_method, num_samples, num_traces, wvlt_cfreq, k, target_snr_db, np.nan, maxit, cc, rre, srer, pes, time_end]

import csv
with open('test_trace.csv', mode='a') as params:
    params_writer = csv.writer(params, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    params_writer.writerow(all_params)

import matplotlib.pyplot as plt
plot_trace = 0
def_figs.figure_trace(dataset_method, y, D_trace, x, y_noisy, x_hat, y_hat, time_axis, plot_trace, name, 1)
plt.close('all')

print('*'*10)
print('ALL DONE')
print('*'*10)
