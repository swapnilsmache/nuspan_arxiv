# %% Testing on Real Data
# O. S. R., dGB Earth Sciences. Penobscot 3D - Survey, 2017. Data retrieved from https://terranubis.com/datainfo/Penobscot

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
parser = argparse.ArgumentParser(description='Testing on Real Data')
parser.add_argument('-b', '--bpi', help='use BPI', action='store_true')
parser.add_argument('-f', '--fista', help='use FISTA', action='store_true')
parser.add_argument('-s', '--sblem', help='use SBL-EM', action='store_true')
parser.add_argument('-n1', '--nuspan1', help='use NuSPAN-1', action='store_true')
parser.add_argument('-n2', '--nuspan2', help='use NuSPAN-2', action='store_true')
args = parser.parse_args()

# %% Model Parameters

wvlt_type = 'ricker'  # 'ricker' or 'bandpass'
wvlt_cfreq = 25       # wavelet central frequency in Hz
num_samples = 300     # number of samples in a trace (must be > 128, length of the Ricker wavelet)
num_traces = 1        # number of seismic traces
k = 0.05              # sparsity factor

## Generate the seismic trace (y), dictionary/kernel matrix (D), reflectivity function (x), and the time axis
zy, D_trace, zx, time_axis, wvlt_t, wvlt_amp = def_models.trace_real(wvlt_type, wvlt_cfreq, num_samples, num_traces, k)

# Load real seismic traces
seis0 = np.load('penobscot_3d.npy')     # shape = [800, 201, 121] , i.e., [num_samples, inlines, xlines]
seis0.astype(np.float32)

num_cols = 201

x_hat = np.empty( [0, num_cols] )
x0_hat = np.empty( [0, num_cols] )
y = np.empty( [0, num_cols] )
y0 = np.empty( [0, num_cols] )
y_hat = np.empty( [0, num_cols] )
y0_hat = np.empty( [0, num_cols] )

time_start = time.time()

xline_0 = 115
xline = 115
while xline < 116:
    row2 = 0
    row1 = 0
    while row1 < 800:
        row2 = row1 + 200
        y0_part = seis0[row1:row2, :, xline]               # original y
        num_samples, num_traces = y0_part.shape

        from sklearn.preprocessing import MaxAbsScaler
        transformer = MaxAbsScaler().fit(y0_part)
        y_part = transformer.transform(y0_part)                  # normalized y

        y_zeros = np.zeros([int( (300-num_samples)/2 ), num_traces])
        y0_part = np.vstack( [y_zeros, y0_part, y_zeros] )
        y_part = np.vstack( [y_zeros, y_part, y_zeros] )
        num_samples, num_traces = y0_part.shape


        ### Input Parameters
        x_hat_part = []
        if args.bpi:
            dataset_method = 'real_bpi'

            lambd = 7.e-2
            maxit = 8
            tol = 1.e-4
            
            for i in range(num_traces):
                x_hat_part.append(def_algorithms.bpi(D_trace, y_part[:,i], num_samples, maxit, lambd, tol))
                if (i+1) % 50 == 0:
                    print('Trace {} done'.format(i+1))
            x_hat_part = np.array(x_hat_part).T

        if args.fista:
            dataset_method = 'real_fista'
            
            lambd = 7.e-2
            maxit = 120
            
            for i in range(num_traces):
                x_hat_part.append(def_algorithms.fista(D_trace, y_part[:,i], lambd, maxit))
                if (i+1) % 100 == 0:
                    print('Trace {} done'.format(i+1))
            x_hat_part = np.array(x_hat_part).T

        if args.sblem:
            dataset_method = 'real_sblem'

            flag1 = 2
            flag3 = 0
            lambd = 2.e-3
            maxit = 60

            x_hat_part = []
            for i in range(num_traces):
                x_hat_part.append(def_algorithms.sbl_em(D_trace, y_part[:,i], lambd, maxit, flag1, flag3))
                if (i+1) % 100 == 0:
                    print('Trace {} done'.format(i+1))
            x_hat_part = np.squeeze( np.array(x_hat_part) ).T

        if args.nuspan1:
            dataset_method = 'real_nuspan1'

            L = scipy.linalg.norm(D_trace, ord=2) ** 2
            m, n = num_samples, num_samples

            lambda1_int = 1.0
            mu1_int = 1.0
            gama1_int = 2
            nu1_int = 1.0
            a1_int = 3

            learning_rate = 1.e-3

            import torch
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = def_algorithms.nuspan1(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, 6)
            model = model.float().to(device)

            from pathlib import Path
            ckp_path = Path('nuspan1_real.pt')
            ## Optimizer, Criterion, and Scheduler
            from torch import optim
            opt = optim.Adam(model.parameters(), lr=learning_rate)
            if ckp_path.is_file():
                model, opt, start_epoch = def_algorithms.load_ckp(ckp_path, model, opt)
                model.eval()
            else:
                print('Enter correct model details')

            y_noisy_t = torch.from_numpy(y_part.T)
            if len(y_part.shape) <= 1:
                y_noisy_t = t_test_t.view(1, -1)
            y_noisy_t = y_noisy_t.float().to(device)

            x_out = np.zeros_like(y_part.T)

            with torch.no_grad():
                x_out = model(y_noisy_t.float())
                if len(y_part.shape) <= 1:
                    x_out = x_out.view(-1)
                x_hat_part = (x_out.cpu().numpy()).T

        if args.nuspan2:
            dataset_method = 'real_nuspan2'

            L = scipy.linalg.norm(D_trace, ord=2) ** 2
            m, n = num_samples, num_samples

            lambda1_int = 1.0
            mu1_int = 1.0
            gama1_int = 2
            nu1_int = 1.0
            a1_int = 3

            learning_rate = 1.e-3

            import torch
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = def_algorithms.nuspan2(m, n, D_trace, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, 22)
            model = model.float().to(device)

            from pathlib import Path
            ckp_path = Path('nuspan2_real.pt')
            ## Optimizer, Criterion, and Scheduler
            from torch import optim
            opt = optim.Adam(model.parameters(), lr=learning_rate)
            if ckp_path.is_file():
                model, opt, start_epoch = def_algorithms.load_ckp(ckp_path, model, opt)
                model.eval()
            else:
                print('Enter correct model details')

            y_noisy_t = torch.from_numpy(y_part.T)
            if len(y_part.shape) <= 1:
                y_noisy_t = t_test_t.view(1, -1)
            y_noisy_t = y_noisy_t.float().to(device)

            x_out = np.zeros_like(y_part.T)

            with torch.no_grad():
                x_out = model(y_noisy_t.float())[0]
                if len(y_part.shape) <= 1:
                    x_out = x_out.view(-1)
                x_hat_part = (x_out.cpu().numpy()).T

        x0_hat_part = transformer.inverse_transform(x_hat_part)

        y_hat_part = []
        for i in range(num_traces):
            y_hat_part.append(np.dot(D_trace, x_hat_part[:,i]))
        y_hat_part = np.array(y_hat_part).T
        y0_hat_part = transformer.inverse_transform(y_hat_part)

        y = np.vstack( [y, y_part[50:250,:]] )
        y_hat = np.vstack( [y_hat, y_hat_part[50:250,:]] )
        y0 = np.vstack( [y0, y0_part[50:250,:]] )
        y0_hat = np.vstack( [y0_hat, y0_hat_part[50:250,:]] )
        x_hat = np.vstack( [x_hat, x_hat_part[50:250,:]] )
        x0_hat = np.vstack( [x0_hat, x0_hat_part[50:250,:]] )

        row1 += 200
    xline += 1

time_end = np.round((time.time() - time_start), 4)
print('Time: ', time_end)
print('-'*50)

### save all arrays (y0, y0_hat, x0_hat)
np.save('{}/y_xline_{}.npy'.format(dataset_method, xline_0), y0.astype(np.float32))
np.save('{}/y_hat_xline_{}.npy'.format(dataset_method, xline_0), y0_hat.astype(np.float32))
np.save('{}/x_hat_xline_{}.npy'.format(dataset_method, xline_0), x0_hat.astype(np.float32))

print('*'*10)
print('ALL DONE')
print('*'*10)
