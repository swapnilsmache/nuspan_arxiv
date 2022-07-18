# %% Training NuSPAN

## Convolutional model: y = Dx + noise
# dimensions of y and x same. m == n

import def_algorithms   # import BPI, FISTA, SBL-EM, NuSPAN-1, NuSPAN-2
import def_models       # import trace/wedge models
import def_figs         # import trace/wedge figures definitions

import numpy as np
import time
import scipy

import argparse
parser = argparse.ArgumentParser(description='Train NuSPAN model')
parser.add_argument('-n1', '--nuspan1', help='train NuSPAN-1', action='store_true')
parser.add_argument('-n2', '--nuspan2', help='train NuSPAN-2', action='store_true')
args = parser.parse_args()

import torch
print('torch version', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# torch.cuda.set_device(0)
# device = torch.device("cuda:0")
print('device', device)

# %% Model Parameters

np.random.seed(0)

wvlt_type = 'ricker'  # 'ricker' or 'bandpass'
wvlt_cfreq = 30       # wavelet central frequency in Hz
num_samples = 300     # number of samples in a seismic trace (must be > 128, length of the Ricker wavelet)
# num_traces = 1        # number of seismic traces
num_traces = 510000   # number of seismic traces
k = 0.05              # sparsity factor
target_snr_db = 10

print('Sparsity factor (k)', k)
print('SNR', target_snr_db)

## Generate the seismic trace (y), dictionary/kernel matrix (D), reflectivity function (x), and the time axis
y0, D_trace, x, time_axis, wvlt_t, wvlt_amp = def_models.trace(wvlt_type, wvlt_cfreq, num_samples, num_traces, k)
print('Data Generated')

## Generate noisy signal (y_noisy)
y_noisy_list = []
for i in range(num_traces):
    y_noisy_list.append(def_models.trace_noisy(y0[:, i], target_snr_db=target_snr_db))
y = np.array(y_noisy_list).T
print('Noisy Data Generated')
print('-'*50)

# np.save('x_training.npy', x)
# np.save('y_training.npy', y0)
# np.save('y_noisy_training.npy', y)

# x = np.load('x_training.npy')
# y0 = np.load('y_training.npy')
# y = np.load('y_noisy_training.npy')
# print('Data Loaded')
# num_traces = y.shape[1]

#%% Input Parameters

# Convert data to tensors
## shape[0] of x and y = num_traces, and shape[1] = num_samples
x_t = torch.from_numpy(x.T)
x_t = x_t.float()
y_t = torch.from_numpy(y.T)
y_t = y_t.float()
D_t = torch.from_numpy(D_trace.T)
D_t = D_t.float()

bs = 200
valid_size = 10000
train_size = num_traces - valid_size
train_ds = def_algorithms.dataset(x_t[:-valid_size, :], y_t[:-valid_size, :])
print('Batch size:', bs)
print('Train size:', train_size)
print('Validation size:', valid_size)

from torch.utils.data import Dataset, DataLoader
train_dl = DataLoader(train_ds, bs, shuffle=True)

L = scipy.linalg.norm(D_trace, ord=2) ** 2
m, n = num_samples, num_samples

if args.nuspan1:
    folder = 'trace_nuspan1'

    lambda1_int = 1.e-2
    mu1_int = 10.0
    gama1_int = 2
    nu1_int = 1.e-1
    a1_int = 3
    learning_rate = 1.e-3
    maxit = 10
    # maxit = 15
    omega1, omega2, omega3 = (1/3), (1/3), (1/3)

    model = def_algorithms.nuspan1(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit)

if args.nuspan2:
    folder = 'trace_nuspan2'

    lambda1_int = 1.e+1
    mu1_int = 1.e-6
    gama1_int = 4
    nu1_int = 2.e-2
    a1_int = 4
    learning_rate = 1.e-3
    maxit = 10
    # maxit = 15

    model = def_algorithms.nuspan2(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit)

print('folder', folder)
from os import path
if not path.exists(folder):
    os.makedirs(folder)

print('lambda1', lambda1_int)
print('mu1', mu1_int)
print('gamma1', gama1_int)
print('nu1', nu1_int)
print('a1', a1_int)
print('learning rate', learning_rate)
print('maxit', maxit)

model = model.float().to(device)
model.weights_init()

## Optimizer and Criterion
from torch import optim
opt = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.SmoothL1Loss()
start_epoch = 0
from pathlib import Path
ckp_path = Path('{}/checkpoint.pt'.format(folder))
if ckp_path.is_file():
    model, opt, start_epoch = def_algorithms.load_ckp(ckp_path, model, opt)
epochs = 100
print('start epoch', start_epoch)
print('epochs', epochs)

import csv
loss_file = '{}/{}_loss.csv'.format(folder, folder)

if start_epoch!=0:
    with open(loss_file, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data_array = np.asarray(data, dtype=np.float32)
    opt_v_loss = min(data_array[:,2])
    opt_idx = np.argmin(data_array[:,2])
    opt_epochs = int(data_array[opt_idx,0])
    opt_t_loss = data_array[opt_idx,1]
    opt_t_supp = data_array[opt_idx,3]
    opt_v_supp = data_array[opt_idx,4]

    print('-'*50)
    print('opt_epochs, opt_t_loss, opt_v_loss, opt_t_supp, opt_v_supp')
    print(opt_epochs, opt_t_loss, opt_v_loss, opt_t_supp, opt_v_supp)
print('-'*50)

### Training
time_train_start = time.time()

t_loss_list = []
t_supp_list = []
v_loss_list = []
v_supp_list = []
opt_v_supp = 0

import copy
print('Epoch, Train Supp, Val Supp, Train Loss, Val Loss')
print('-'*50)
is_best = False

for epoch in range(epochs):
    is_best = False
    model.train()
    t_loss_total = 0
    t_supp_total = 0
    v_loss_total = 0
    v_supp_total = 0
    
    for xb, yb in train_dl:
        # opt.zero_grad()
        if args.nuspan1:
            xb_hat = model(yb.float())      # compute the outputs
        if args.nuspan2:
            xb_hat = model(yb.float())[0]      # compute the outputs

        # compute the support metric
        with torch.no_grad():
            for i in range(bs):
                t_supp = def_algorithms.support_metric(xb[i, :].detach().cpu().numpy(), xb_hat[i, :].detach().cpu().numpy())
                t_supp_total = t_supp_total + t_supp

        # compute the loss
        t_loss = loss_func(xb_hat.float(), xb.float()) / bs
        t_loss_total += t_loss.detach().cpu().data
        opt.zero_grad()
        t_loss.backward()               # to propagate gradients backward
        opt.step()                      # to take forward step and optimize parameters

    model.eval()
    # compute the support metric
    with torch.no_grad():
        t_supp_list.append(t_supp_total / (train_size - valid_size))
        t_loss_list.append(t_loss_total.detach().data)

        ### Validation
        ybv = y_t[train_size:, :].to(device)
        xbv = x_t[train_size:, :].to(device)
        if args.nuspan1:
            xbv_hat = model(ybv.float())
        if args.nuspan2:
            xbv_hat = model(ybv.float())[0]
        v_loss = loss_func(xbv_hat.float(), xbv.float()) / valid_size
        v_loss_list.append(v_loss.data)

        for i in range(valid_size):
            v_supp = def_algorithms.support_metric(xbv[i, :].detach().cpu().numpy(), xbv_hat[i, :].detach().cpu().numpy())
            v_supp_total = v_supp_total + v_supp
        v_supp_list.append(v_supp_total/valid_size)
    
    if start_epoch + epoch == 0:
        opt_v_loss = v_loss_list[0].item()
        opt_t_loss = t_loss_list[0].item()
        opt_model_wts = copy.deepcopy(model.state_dict())
        opt_epochs = start_epoch + epoch + 1
        is_best = True
    else:
        if v_loss_list[epoch].item() < opt_v_loss:
            opt_v_loss = v_loss_list[epoch].item()
            opt_t_loss = t_loss_list[epoch].item()
            opt_model_wts = copy.deepcopy(model.state_dict())
            opt_epochs = start_epoch + epoch + 1
            is_best = True

    if (start_epoch + epoch + 1) % 10 == 0:
        print((start_epoch + epoch + 1), np.round(t_supp_list[epoch], 4), np.round(v_supp_list[epoch], 4), np.round(t_loss_total.item(), 10), np.round(v_loss.item(), 10))
    if (start_epoch + epoch + 1) % 10 == 0:
        ## Figure 2: Training and Validation Loss
        def_figs.figure_params(folder, t_supp_list, v_supp_list, num_traces, start_epoch + epoch + 1, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, 'Support', 1)
        ### Figure 3: Training and Validation Support
        def_figs.figure_params(folder, t_loss_list, v_loss_list, num_traces, start_epoch + epoch + 1, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, 'Loss', 2)

    checkpoint = {
    'epoch': start_epoch + epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': opt.state_dict()
    }
    def_algorithms.save_ckp(folder, checkpoint, is_best)

    all_losses = [ start_epoch + epoch + 1, np.round(t_loss_total.item(), 15), np.round(v_loss.item(), 15), np.round(t_supp_list[epoch], 4), np.round(v_supp_list[epoch], 4) ]
    with open(loss_file, mode='a') as params:
        params_writer = csv.writer(params, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        params_writer.writerow(all_losses)

with open(loss_file, newline='') as csvfile:
        data = list(csv.reader(csvfile))
data_array = np.asarray(data, dtype=np.float32)
def_figs.figure_params(folder, data_array[:,1], data_array[:,2], num_traces, start_epoch+epochs, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, 'Overall Loss', num=3)
def_figs.figure_params(folder, data_array[:,3], data_array[:,4], num_traces, start_epoch+epochs, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, 'Overall Support', num=4)

print('-'*50)
print('opt_epochs', opt_epochs, 'opt_t_loss', np.round(opt_t_loss, 4), 'opt_v_loss', np.round(opt_v_loss, 10))
print('-'*50)

if is_best == True:
    model.load_state_dict(opt_model_wts)
    tot_epochs = start_epoch + epochs
else:
    ckp_path = '{}/best_model.pt'.format(folder)
    model, opt, start_epoch = def_algorithms.load_ckp(ckp_path, model, opt)
    tot_epochs = start_epoch

### Load and evaluate best model
model.eval()
with torch.no_grad():
    if args.nuspan1:
        x_out = model(y_t.float())
    if args.nuspan2:
        x_out, learned_omegas = model(y_t.float())

        omega1 = np.mean(learned_omegas.cpu().numpy()[:,:,0])
        omega2 = np.mean(learned_omegas.cpu().numpy()[:,:,1])
        omega3 = np.mean(learned_omegas.cpu().numpy()[:,:,2])

    if len(y.shape) <= 1:
        x_out = x_out.view(-1)
    x_hat = (x_out.cpu().numpy()).T

    lambda1_learned = np.round(np.mean( model.lambda1.detach().cpu().numpy().copy() ), 4)
    mu1_learned = np.round(np.mean( model.lambda2.detach().cpu().numpy().copy() ), 4)
    gama1_learned = np.round(np.mean( model.gama.detach().cpu().numpy().copy() ), 4)
    nu1_learned = np.round(np.mean( model.lambda3.detach().cpu().numpy().copy() ), 4)
    a1_learned = np.round(np.mean( model.a.detach().cpu().numpy().copy() ), 4)

time_train_end = time.time() - time_train_start
print('Training Time: ', np.round(time_train_end, 4))

### Training and Validation Metrics
y_hat = []
for i in range(num_traces):
    y_hat.append( np.dot(D_trace, x_hat[:, i]) )
y_hat = np.array(y_hat).T

cc_train = []
rre_train = []
srer_train = []
pes_train = []

for i in range(num_traces):
    cc_train.append(np.abs(np.corrcoef(x[:, i], x_hat[:, i])[0, 1]))

    error_num = scipy.linalg.norm(x[:, i]) ** 2
    error_den = scipy.linalg.norm(x_hat[:, i] - x[:, i]) ** 2
    rre_train.append(error_den / error_num)
    srer_train.append(10 * np.log10(error_num / error_den))
    
    pes_train.append(def_algorithms.pes(x[:, i], x_hat[:, i]))

print('-'*50)
print('Training Metrics')
print('-'*50)
cc_train = np.round( np.nanmean(np.array(cc_train)), 4 )
print('Correlation Coefficient (Reflectivity): {}'.format(cc_train))

rre_train = np.round( np.nanmean(np.array(rre_train)), 4 )
print('Relative Reconstruction Error: {}'.format(rre_train))

srer_train = np.round( np.nanmean(np.array(srer_train)), 4 )
print('Signal-to-Reconstruction Error Ratio: {} dB'.format(srer_train))

pes_train = np.round( np.nanmean(np.array(pes_train)), 4 )
print('Probability of Error in Support: {}'.format(pes_train))
print('-'*50)

cc_val = []
rre_val = []
srer_val = []
pes_val = []

for i in range(train_size, num_traces):
    cc_val.append(np.abs(np.corrcoef(x[:, i], x_hat[:, i])[0, 1]))

    error_num = scipy.linalg.norm(x[:, i]) ** 2
    error_den = scipy.linalg.norm(x_hat[:, i] - x[:, i]) ** 2
    rre_val.append(error_den / error_num)
    srer_val.append(10 * np.log10(error_num / error_den))
    
    pes_val.append(def_algorithms.pes(x[:, i], x_hat[:, i]))

print('-'*50)
print('Validation Metrics')
print('-'*50)
cc_val = np.round( np.nanmean(np.array(cc_val)), 4 )
print('Correlation Coefficient (Reflectivity): {}'.format(cc_val))

rre_val = np.round( np.nanmean(np.array(rre_val)), 4 )
print('Relative Reconstruction Error: {}'.format(rre_val))

srer_val = np.round( np.nanmean(np.array(srer_val)), 4 )
print('Signal-to-Reconstruction Error Ratio: {} dB'.format(srer_val))

pes_val = np.round( np.nanmean(np.array(pes_val)), 4 )
print('Probability of Error in Support: {}'.format(pes_val))
print('-'*50)

### Testing
## Generate testing data
np.random.seed(4)
num_traces_test = 1000      # number of seismic traces for testing

## Generate the seismic trace (y), dictionary/kernel matrix (D), reflectivity function (x), and the time axis
y0_test, D_trace_test, x_test, time_axis_test, wvlt_t_test, wvlt_amp_test = def_models.trace(wvlt_type, wvlt_cfreq, num_samples, num_traces_test, k)

# Generate noisy signal (y_noisy)
y_test_noisy_list = []
for i in range(num_traces_test):
    y_test_noisy_list.append( def_models.trace_noisy(y0_test[:, i], target_snr_db=target_snr_db) )
y_test = np.array(y_test_noisy_list).T

time_test_start = time.time()

y_test_t = torch.from_numpy(y_test.T)
if len(y_test.shape) <= 1:
    y_test_t = t_test_t.view(1, -1)
y_test_t = y_test_t.float().to(device)

x_out_test = np.zeros_like(y_test.T)
model.eval()
with torch.no_grad():
    if args.nuspan1:
        x_out_test = model(y_test_t.float())
    if args.nuspan2:
        x_out_test = model(y_test_t.float())[0]    
    if len(y_test.shape) <= 1:
        x_out_test = x_out_test.view(-1)
    x_hat_test = (x_out_test.cpu().numpy()).T

time_test_end = time.time() - time_test_start
print('Testing Time: ', np.round(time_test_end, 4))
print('-'*50)

y_hat_test = []
for i in range(num_traces_test):
    # y_hat_test_trace = np.dot(D_trace_test, x_hat_test[:,i])
    y_hat_test.append( np.dot(D_trace_test, x_hat_test[:,i]) )
y_hat_test = np.array(y_hat_test).T

cc_test = []
rre_test = []
srer_test = []
pes_test = []

for i in range(num_traces_test):
    cc_test.append(np.abs(np.corrcoef(x[:, i], x_hat[:, i])[0, 1]))

    error_num = scipy.linalg.norm(x[:, i]) ** 2
    error_den = scipy.linalg.norm(x_hat[:, i] - x[:, i]) ** 2
    rre_test.append(error_den / error_num)
    srer_test.append(10 * np.log10(error_num / error_den))
    
    pes_test.append(def_algorithms.pes(x[:, i], x_hat[:, i]))

print('-'*50)
print('Testing Metrics')
print('-'*50)
cc_test = np.round( np.nanmean(np.array(cc_test)), 4 )
print('Correlation Coefficient (Reflectivity): {}'.format(cc_test))

rre_test = np.round( np.nanmean(np.array(rre_test)), 4 )
print('Relative Reconstruction Error: {}'.format(rre_test))

srer_test = np.round( np.nanmean(np.array(srer_test)), 4 )
print('Signal-to-Reconstruction Error Ratio: {} dB'.format(srer_test))

pes_test = np.round( np.nanmean(np.array(pes_test)), 4 )
print('Probability of Error in Support: {}'.format(pes_test))
print('-'*50)

### Params and Figures
all_params = [folder, num_samples, num_traces, wvlt_cfreq, k, target_snr_db, np.nan, maxit, 
            tot_epochs, learning_rate, np.round(time_train_end, 4), np.round(time_test_end, 4), 
            opt_epochs, np.round(opt_t_loss, 4), np.round(opt_v_loss, 15), 
            omega1, lambda1_int, lambda1_learned, 
            omega2, mu1_int, mu1_learned, gama1_int, gama1_learned, 
            omega3, nu1_int, nu1_learned, a1_int, a1_learned, 
            cc_train, cc_val, cc_test, 
            rre_train, rre_val, rre_test, 
            srer_train, srer_val, srer_test, 
            pes_train, pes_val, pes_test
            ]

import csv
params_file = '{}/{}.csv'.format(folder, folder)
with open(params_file, mode='a') as params:
    params_writer = csv.writer(params, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    params_writer.writerow(all_params)

idx = 0 # idx = np.random.randint(0, high = num_traces_test)
### Training: Reconstructed vs. True (Reflectivity and Seismic Trace)
def_figs.figure_trace_train(folder, y0, D_trace, x, y, x_hat, y_hat, time_axis, num_traces, idx, opt_epochs, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, 'train', 5)
### Testing: Reconstructed vs. True (Reflectivity and Seismic Trace)
def_figs.figure_trace_train(folder, y0_test, D_trace_test, x_test, y_test, x_hat_test, y_hat_test, time_axis_test, num_traces_test, idx, opt_epochs, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, 'test', 6)

print('*'*10)
print('ALL DONE')
print('*'*10)





