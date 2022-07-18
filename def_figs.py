### Define figures

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
font = {'size' : 8}
matplotlib.rc('font', **font)

### Plot Synthetic 1-D trace
def figure_trace(folder, y, D_trace, x, y_noisy, x_hat, y_hat, time_axis, plot_trace, name, num=1):

    # Converting zeros to nan for plotting
    x_nan = x.copy()
    x_nan [ x_nan==0 ] = np.nan
    x_hat_nan = x_hat.copy()
    x_hat_nan [ x_hat_nan==0 ] = np.nan

    fig_trace = plt.figure(num=num, figsize=(6.75, 6.75))
    fig_trace.set_facecolor('white')
    gs_rc = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax_r1 = fig_trace.add_subplot(gs_rc[0])
    ax_r1.plot(time_axis, y[:, plot_trace], color='r', label='Seismic trace (True)', linewidth=5, alpha=0.5)
    ax_r1.plot(time_axis, y_hat[:, plot_trace], color='g', label='Seismic trace (Predicted)', linewidth=2)
    y_lim1 = 1.1*np.max(np.abs(y[:, plot_trace]))
    ax_r1.set_ylim(-y_lim1, y_lim1)
    ax_r1.grid()
    ax_r1.set_xlabel('Time (ms)')
    ax_r1.set_ylabel('Amplitude')
    ax_r1.legend()
    ax_r1.set_title(name)

    ax_r2 = fig_trace.add_subplot(gs_rc[1])
    markerline1, stemlines1, baseline1 = ax_r2.stem(time_axis, x_nan[:, plot_trace], markerfmt='r ', linefmt='r', label='Reflectivity (True)', use_line_collection=True)
    markerline3, stemlines3, baseline3 = ax_r2.stem(time_axis, x_hat_nan[:, plot_trace], markerfmt='g ', linefmt='g', label='Reflectivity (Predicted)', use_line_collection=True)
    plt.setp(stemlines1, linewidth=5, alpha=0.5)
    plt.setp(stemlines3, linewidth=2)
    plt.setp(baseline1, color='k', linewidth=1)
    plt.setp(baseline3, color='k', linewidth=1)
    markerline1.set_markerfacecolor('none')
    ax_r2.legend()
    y_lim2 = 1.1*np.max(np.abs(x[:, plot_trace]))
    ax_r2.set_ylim(-y_lim2, y_lim2)
    ax_r2.grid()
    ax_r2.set_xlabel('Time (ms)')
    ax_r2.set_ylabel('Amplitude')

    gs_rc.tight_layout(fig_trace)

    plt.savefig('{}_num_{}.pdf'.format(folder, str(plot_trace)))

def figure_trace_train(folder, y, D_trace, x, y_noisy, x_hat, y_hat, time_axis, num_traces, plot_trace, epochs, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, name, num=1):

    # Converting zeros to nan for plotting
    x_nan = x.copy()
    x_nan [ x_nan==0 ] = np.nan
    x_hat_nan = x_hat.copy()
    x_hat_nan [ x_hat_nan==0 ] = np.nan

    fig_trace = plt.figure(num=num, figsize=(18, 12))
    fig_trace.set_facecolor('white')
    gs_rc = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax_r1 = fig_trace.add_subplot(gs_rc[0])
    ax_r1.plot(time_axis, y[:, plot_trace], color='r', label='Seismic trace_True', linewidth=5, alpha=0.5)
    ax_r1.plot(time_axis, y_hat[:, plot_trace], color='g', label='Seismic trace_Predicted', linewidth=2)
    y_lim1 = 1.1*np.max(np.abs(y[:, plot_trace]))
    ax_r1.set_ylim(-y_lim1, y_lim1)
    ax_r1.grid()
    ax_r1.set_xlabel('Time (ms)')
    ax_r1.set_ylabel('Amplitude')
    ax_r1.legend()
    ax_r1.set_title(name)
    
    ax_r2 = fig_trace.add_subplot(gs_rc[1])
    markerline1, stemlines1, baseline1 = ax_r2.stem(time_axis, x_nan[:, plot_trace], markerfmt='r ', linefmt='r', label='Ref_True')#, use_line_collection=True)
    markerline3, stemlines3, baseline3 = ax_r2.stem(time_axis, x_hat_nan[:, plot_trace], markerfmt='g ', linefmt='g', label='Ref_Predicted')#, use_line_collection=True)
    plt.setp(stemlines1, linewidth=5, alpha=0.5)
    plt.setp(stemlines3, linewidth=2)
    plt.setp(baseline1, color='k', linewidth=1)
    plt.setp(baseline3, color='k', linewidth=1)
    markerline1.set_markerfacecolor('none')
    ax_r2.legend()
    y_lim2 = 1.1*np.max(np.abs(x[:, plot_trace]))
    ax_r2.set_ylim(-y_lim2, y_lim2)
    ax_r2.grid()
    ax_r2.set_xlabel('Time (ms)')
    ax_r2.set_ylabel('Amplitude')
    
    gs_rc.tight_layout(fig_trace)
    plt.savefig('{}/trace_{}_numtraces_{}_maxit_{}_epch_{}_lr_{}_L1_{}_L2_{}_G_{}_L3_{}_a_{}.png'.format(folder, name, str(num_traces), str(maxit), str(epochs), str(learning_rate), str(lambda1_int), str(mu1_int), str(gama1_int), str(nu1_int), str(a1_int) ))
    plt.close('all')


### Plot Training and Validation Loss/Support (for NuSPAN)
def figure_params(folder, train_param, valid_param, num_traces, epochs, maxit, learning_rate, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, paramname, num=2):
    fig_supp, ax_supp = plt.subplots(num=num, ncols=2, nrows=1, constrained_layout=True, figsize=(15,6))
    fig_supp.set_facecolor('white')
    ax_supp[0].plot(train_param)
    ax_supp[0].set_title('Training {}'.format(paramname))
    ax_supp[0].set_xlabel('Epochs')
    ax_supp[1].plot(valid_param)
    ax_supp[1].set_title('Validation {}'.format(paramname))
    ax_supp[1].set_xlabel('Epochs')
    plt.savefig('{}/{}_numtraces_{}_maxit_{}_epch_{}_lr_{}_L1_{}_m1_{}_g1_{}_n1_{}_a1_{}.png'.format(folder, paramname, str(num_traces), str(maxit), str(epochs), str(learning_rate), str(lambda1_int), str(mu1_int), str(gama1_int), str(nu1_int), str(a1_int) ))
    plt.close('all')

### Plot wedge model
def figure_wedge(folder, x_axis, time_axis, model_type, name, num=1):

    fig_ref, ax_ref = plt.subplots(ncols=1, nrows=1, constrained_layout=True, num=num, figsize=(3.5, 2))
    fig_ref.set_facecolor('white')

    plot_vawig(ax_ref, x_axis, time_axis, excursion=1)
    ax_ref.set_ylim((20, 130))
    ax_ref.invert_yaxis()
    ax_ref.set_xlabel('WEDGE THICKNESS (m)')
    ax_ref.set_ylabel('TIME (ms)')
    # ax_ref.set_title(name)
    plt.savefig('{}_{}.pdf'.format(model_type, folder), dpi=600)


# Modified from original code by Wes Hamlyn (https://github.com/seg/tutorials-2014/blob/master/1412_Tuning_and_AVO/tuning_prestack.py)
# Reference: Hamlyn, W. (2014). Thin beds, tuning, and AVO. The Leading Edge, 33(12), 1394-1396. https://doi.org/10.1190/tle33121394.1

def plot_vawig(axhdl, data, t, excursion, highlight=None):

    import numpy as np
    import matplotlib.pyplot as plt

    [ntrc, nsamp] = data.shape
    
    # t = np.hstack([0, t, t.max()])
    
    for i in range(0, ntrc):
        tbuf = excursion * data[i] / 0.75 + i
        # tbuf = excursion * data[i] / np.max(np.abs(data)) + i
        # tbuf = tbuf*1000
        # tbuf = np.hstack([i, tbuf, i])
            
        if i==highlight:
            lw = 0.2
        else:
            lw = 0.2

        axhdl.plot(tbuf, t, color='black', linewidth=lw)

        plt.fill_betweenx(t, tbuf, i, where=data[i]<=0, facecolor='b', linewidth=0) # [0.6,0.6,1.0]
        plt.fill_betweenx(t, tbuf, i, where=data[i]>=0, facecolor='r', linewidth=0) # [1.0,0.7,0.7]
        
    axhdl.set_xlim((-excursion*1.2, ntrc-1+excursion*1.2))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()
    return t,tbuf,i

def plot_nofill(axhdl, data, t, excursion, highlight=None):

    import numpy as np
    import matplotlib.pyplot as plt

    [ntrc, nsamp] = data.shape
    
    # t = np.hstack([0, t, t.max()])
    
    for i in range(0, ntrc):
        tbuf = excursion * data[i] / np.max(np.abs(data)) + i
        # tbuf = tbuf*1000
        # tbuf = np.hstack([i, tbuf, i])
            
        if i==highlight:
            lw = 2
        else:
            lw = 0.5

        p1 = axhdl.plot(tbuf, t, color='black', linewidth=lw, linestyle=':')

        # plt.fill_betweenx(t, tbuf, i, where=data[i]<=0, facecolor=[0.6,0.6,1.0], linewidth=0)
        # plt.fill_betweenx(t, tbuf, i, where=data[i]>=0, facecolor=[1.0,0.7,0.7], linewidth=0)
        
    axhdl.set_xlim((-excursion, ntrc+excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()
    return p1



