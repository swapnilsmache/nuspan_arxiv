# %% Testing on the Simulated 2-D Marmousi2 model
# https://wiki.seg.org/wiki/AGL_Elastic_Marmousi
# To download, click the link or paste the line below into terminal:
# wget https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz

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
parser = argparse.ArgumentParser(description='Testing on the Simulated 2-D Marmousi2 model')
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
num_traces = 1        # number of seismic traces
k = 0.05              # sparsity factor

## Generate the seismic trace (y), dictionary/kernel matrix (D), reflectivity function (x), and the time axis
zy, D_trace, zx, time_axis, wvlt_t, wvlt_amp = def_models.trace_mm(wvlt_type, wvlt_cfreq, num_samples, num_traces, k)

# Load Marmousi2 model reflectivity
import shutil
shutil.unpack_archive('marmousi2_reflectivity.npy.zip')
rc = np.load('marmousi2_reflectivity.npy')
rc[np.abs(rc) < 0.01*np.max(np.abs(rc))] = 0.0  # muting low-amplitude spikes (<1% of the absolute of the maximum amplitude)
# rc.astype(np.float32)

num_cols = 1000         # Change to 13601 to evaluate complete Marmousi2 model (total number of traces in the Marmousi2 model)

x0 = np.empty( [0, num_cols] )
x0_hat = np.empty( [0, num_cols] )
y0 = np.empty( [0, num_cols] )
y0_noisy = np.empty( [0, num_cols] )
y0_hat = np.empty( [0, num_cols] )

time_start = time.time()

row1_0 = 400
row1 = row1_0
col1_0 = 2000
col1 = col1_0
while col1 < 3000:
    col2 = col1 + 1000
    row2 = 0
    while row1 < 1400:

### For complete Marmousi2 model, comment out above loop section and uncomment below lop section
### Also change num_cols on line 37 to 13601 (total number of traces in the Marmousi2 model)
# row1_0 = 200
# row1 = row1_0
# col1_0 = 0
# col1 = col1_0
# while col1 < num_cols:
#     col2 = col1 + num_cols
#     row2 = 0
#     while row1 < 2800:

        row2 = row1 + 200
        print('-'*10)
        print('row1', row1)
        print('row2', row2)
        print('col1', col1)
        print('col2', col2)
        print('-'*10)
        x0_part = rc[row1:row2, col1:col2]                      # original x
        num_samples, num_traces = x0_part.shape

        from sklearn.preprocessing import MaxAbsScaler
        transformer = MaxAbsScaler().fit(x0_part)
        x = transformer.transform(x0_part)                      # normalized x

        x_zeros = np.zeros([int( (300-num_samples)/2 ), num_traces])
        x0_part = np.vstack( [x_zeros, x0_part, x_zeros] )
        x = np.vstack( [x_zeros, x, x_zeros] )
        num_samples, num_traces = x0_part.shape

        y = []
        y0_part = []
        for i in range(num_traces):
            y0_part.append( np.dot(D_trace, x0_part[:,i]) )
            y.append( np.dot(D_trace, x[:,i]) )
        y0_part = np.array(y0_part).T                           # original y
        y = np.array(y).T                                       # normalized y

        # Generate noisy signal (y_noisy)
        target_snr_db = 10
        y_noisy_list = []
        y0_noisy_list = []
        for i in range(num_traces):
            y_noisy_list.append(def_models.trace_noisy(y[:, i], target_snr_db=target_snr_db))
            y0_noisy_list.append(def_models.trace_noisy(y0_part[:, i], target_snr_db=target_snr_db))
        y0_noisy_part = np.array(y0_noisy_list).T               # noisy original y
        y_noisy = np.array(y_noisy_list).T                      # noisy normalized y

        ### Input Parameters
        x_hat_part = []
        if args.bpi:
            dataset_method = 'marmousi2_bpi'

            lambd = 0.1
            maxit = 7
            tol = 1.e-4

            
            for i in range(num_traces):
                x_hat_part.append(def_algorithms.bpi(D_trace, y_noisy[:,i], num_samples, maxit, lambd, tol))
                if (i+1) % 100 == 0:
                    print('Trace {} done'.format(i+1))
            x_hat_part = np.array(x_hat_part).T

        if args.fista:
            dataset_method = 'marmousi2_fista'
            
            lambd = 2.e-1
            maxit = 120
            
            for i in range(num_traces):
                x_hat_part.append(def_algorithms.fista(D_trace, y_noisy[:,i], lambd, maxit))
                if (i+1) % 100 == 0:
                    print('Trace {} done'.format(i+1))
            x_hat_part = np.array(x_hat_part).T

        if args.sblem:
            dataset_method = 'marmousi2_sblem'

            flag1 = 2
            flag3 = 0
            lambd = 2.e-2
            maxit = 50

            x_hat_part = []
            for i in range(num_traces):
                x_hat_part.append(def_algorithms.sbl_em(D_trace, y_noisy[:,i], lambd, maxit, flag1, flag3))
                if (i+1) % 100 == 0:
                    print('Trace {} done'.format(i+1))
            x_hat_part = np.squeeze( np.array(x_hat_part) ).T

        if args.bpi or args.fista or args.sblem:
            x0_hat_part = transformer.inverse_transform(x_hat_part)

        if args.nuspan1:
            dataset_method = 'marmousi2_nuspan1'

            L = scipy.linalg.norm(D_trace, ord=2) ** 2
            m, n = num_samples, num_samples

            lambda1_int = 1.0
            mu1_int = 1.0
            gama1_int = 2
            nu1_int = 1.0
            a1_int = 3

            learning_rate = 1.e-3

            maxit = 5
            import torch
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = def_algorithms.nuspan1(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit)
            model = model.float().to(device)

            from pathlib import Path
            ckp_path = Path('nuspan1_marmousi2.pt'.format(maxit))
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

            D_trace_t = torch.from_numpy(D_trace.T)
            D_trace_t = D_trace_t.float().to(device)

            x_t = torch.from_numpy(x.T)
            x_out = np.zeros_like(x_t)

            with torch.no_grad():
                x_out = model(y_noisy_t.float())
                if len(y_noisy.shape) <= 1:
                    x_out = x_out.view(-1)
                x_hat_part = (x_out.cpu().numpy()).T

            x0_hat_part = transformer.inverse_transform(x_hat_part)

            # ### Re-estimating the amplitudes over the supports given by NuSPAN-1 (comment out to see results without re-estimating the amplitudes)
            # x0_hat_pinv = np.zeros_like(x0_hat_part)
            # x0_hat_pinv = []
            # for i in range(num_traces):
            #     x_hat_nnz = np.nonzero(x0_hat_part[:,i])
            #     H = D_trace.copy()
            #     H = H[:, np.r_[x_hat_nnz]]
            #     H_inv = np.linalg.pinv(H)

            #     amplitudes = np.dot(H_inv, y0_noisy_part[:,i])
            #     amplitudes = np.array(amplitudes).T

            #     x_hat_pinv_trace = x0_hat_part[:,i].copy()
            #     x_hat_pinv_trace[np.r_[x_hat_nnz]] = amplitudes[:]
            #     if any(np.abs(x_hat_pinv_trace - x0_hat_part[:,i]) >= 0.0005):
            #         x_hat_pinv_trace = x0_hat_part[:,i]
            #     x0_hat_pinv.append(x_hat_pinv_trace)
            # x0_hat_pinv = np.array(x0_hat_pinv).T
            # x0_hat_part = x0_hat_pinv.copy()
            # ### (comment out upto here to see results without re-estimating the amplitudes)

        if args.nuspan2:
            dataset_method = 'marmousi2_nuspan2'

            L = scipy.linalg.norm(D_trace, ord=2) ** 2
            m, n = num_samples, num_samples

            lambda1_int = 1.0
            mu1_int = 1.0
            gama1_int = 2
            nu1_int = 1.0
            a1_int = 3

            learning_rate = 1.e-3

            maxit = 21
            import torch
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = def_algorithms.nuspan2(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit)
            model = model.float().to(device)

            from pathlib import Path
            ckp_path = Path('nuspan2_marmousi2.pt')
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

            D_trace_t = torch.from_numpy(D_trace.T)
            D_trace_t = D_trace_t.float().to(device)

            x_t = torch.from_numpy(x.T)
            x_out = np.zeros_like(x_t)

            with torch.no_grad():
                x_out = model(y_noisy_t.float())[0]
                if len(y_noisy.shape) <= 1:
                    x_out = x_out.view(-1)
                x_hat_part = (x_out.cpu().numpy()).T

            x0_hat_part = transformer.inverse_transform(x_hat_part)

            # ### Re-estimating the amplitudes over the supports given by NuSPAN-1 (comment out to see results without re-estimating the amplitudes)
            # x0_hat_pinv = np.zeros_like(x0_hat_part)
            # x0_hat_pinv = []
            # for i in range(num_traces):
            #     x_hat_nnz = np.nonzero(x0_hat_part[:,i])
            #     H = D_trace.copy()
            #     H = H[:, np.r_[x_hat_nnz]]
            #     H_inv = np.linalg.pinv(H)

            #     amplitudes = np.dot(H_inv, y0_noisy_part[:,i])
            #     amplitudes = np.array(amplitudes).T

            #     x_hat_pinv_trace = x0_hat_part[:,i].copy()
            #     x_hat_pinv_trace[np.r_[x_hat_nnz]] = amplitudes[:]
            #     if any(np.abs(x_hat_pinv_trace - x0_hat_part[:,i]) >= 0.0005):
            #         x_hat_pinv_trace = x0_hat_part[:,i]
            #     x0_hat_pinv.append(x_hat_pinv_trace)
            # x0_hat_pinv = np.array(x0_hat_pinv).T
            # x0_hat_part = x0_hat_pinv.copy()
            # ### (comment out upto here to see results without re-estimating the amplitudes)

        y0_hat_part = []
        for i in range(num_traces):
            y0_hat_part.append(np.dot(D_trace, x0_hat_part[:,i]))
        y0_hat_part = np.array(y0_hat_part).T

        x0 = np.vstack( [x0, x0_part[50:250,:]] )
        x0_hat = np.vstack( [x0_hat, x0_hat_part[50:250,:]] )
        y0 = np.vstack( [y0, y0_part[50:250,:]] )
        y0_noisy = np.vstack( [y0_noisy, y0_noisy_part[50:250,:]] )
        y0_hat = np.vstack( [y0_hat, y0_hat_part[50:250,:]] )
        row1 += 200
    col1 += num_cols

time_end = np.round((time.time() - time_start), 4)
num_samples, num_traces = x0.shape

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)
ax0 = axs[0]
vlim = 0.8*max( np.abs(np.max(x0)), np.abs(np.min(x0)) )
pcm0 = ax0.imshow(x0, cmap='seismic', vmin=-vlim, vmax=vlim)
fig.colorbar(pcm0,ax=ax0, fraction=0.046, pad=0.04)

ax1 = axs[1]
# vlim1 = max( np.abs(np.max(x0_hat)), np.abs(np.min(x0_hat)) )
pcm1 = ax1.imshow(x0_hat, cmap='seismic', vmin=-vlim, vmax=vlim)
fig.colorbar(pcm1,ax=ax1, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('{}_row_{}_{}_col_{}_{}.pdf'.format(dataset_method, row1_0, row2, col1_0, col2), bbox_inches='tight', pad_inches = 0, dpi=600)
plt.close('all')

### save all arrays (x, x0_hat, y, y_noisy, y0_hat)
# np.save('{}/x_row_{}_{}_col_{}_{}.npy'.format(dataset_method, row1_0, row2, col1_0, col2), x.astype(np.float32))
# np.save('{}/x_hat_row_{}_{}_col_{}_{}.npy'.format(dataset_method, row1_0, row2, col1_0, col2), x0_hat.astype(np.float32))
# np.save('y_row_{}_{}_col_{}_{}.npy'.format(row1_0, row2, col1_0, col2), y)
# np.save('y_noisy_row_{}_{}_col_{}_{}.npy'.format(row1_0, row2, col1_0, col2), y_noisy)
# np.save('y_hat_row_{}_{}_col_{}_{}.npy'.format(row1_0, row2, col1_0, col2), y0_hat)

cc = []
rre = []
srer = []
pes = []

for i in range(num_traces):
    cc.append(np.abs(np.corrcoef(x0[:,i], x0_hat[:,i])[0, 1]))

    error_num = scipy.linalg.norm(x0[:,i]) ** 2
    error_den = scipy.linalg.norm(x0_hat[:,i] - x0[:,i]) ** 2
    rre.append(error_den / error_num)
    srer.append(10 * np.log10(error_num / error_den))
    
    pes.append(def_algorithms.pes(x0[:,i], x0_hat[:,i]))

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
    all_params = [dataset_method, num_samples, num_traces, wvlt_cfreq, np.nan, target_snr_db, lambd, maxit, cc, rre, srer, pes, time_end, row1_0, row2, col1_0, col2]
if args.nuspan1 or args.nuspan2:
    all_params = [dataset_method, num_samples, num_traces, wvlt_cfreq, np.nan, target_snr_db, np.nan, maxit, cc, rre, srer, pes, time_end, row1_0, row2, col1_0, col2]

import csv
with open('test_marmousi2.csv', mode='a') as params:
    params_writer = csv.writer(params, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    params_writer.writerow(all_params)

print('*'*10)
print('ALL DONE')
print('*'*10)
