#%%
xline = 115

import numpy as np

y = np.load('real_fista/y_xline_{}.npy'.format(xline))
x_bpi = np.load('real_bpi/x_hat_xline_{}.npy'.format(xline))
x_sbl = np.load('real_sblem/x_hat_xline_{}.npy'.format(xline))
x_fista = np.load('real_fista/x_hat_xline_{}.npy'.format(xline))
x_nuspan1 = np.load('real_nuspan1/x_hat_xline_{}.npy'.format(xline))
x_nuspan2 = np.load('real_nuspan2/x_hat_xline_{}.npy'.format(xline))

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use("TkAgg")
font = {'size' : 6}
matplotlib.rc('font', **font)
csfont = {'fontname':'Times New Roman'}

### Load the seismic and reflectivity profiles for well L-30
# Reference: Bianco, E. Geophysical tutorial: well-tie calculus. The Leading Edge, 33(6):674–677, 2014.
# Functions modified from original ones at https://github.com/seg/tutorials-2014/tree/master/1406_Make_a_synthetic

from las import LASReader
L30 = LASReader('L-30.las', null_subs=np.nan)

def f2m(item_in_feet):
    "converts feet to meters"
    try:
        return item_in_feet / 3.28084
    except TypeError:
        return float(item_in_feet) / 3.28084
    return converted

z = f2m(L30.data['DEPTH'])      # convert feet to metres
DT = L30.data['DT']*3.28084     # convert usec/ft to usec/m
RHOB = L30.data['RHOB']*1000    # convert to SI units

scaled_dt = 0.1524 * np.nan_to_num(DT) / 1e6
tcum = 2 * np.cumsum(scaled_dt)
log_start_time = 0.410498125651
tdr = tcum + log_start_time

Z = (1e6/DT) * RHOB
RC = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])

# RESAMPLING FUNCTION
dt = 0.004
maxt = 3.0
t = np.arange(0, maxt, dt) 
Z_t = np.interp(x = t, xp = tdr, fp = Z)
RC_t = (Z_t[1:] - Z_t[:-1]) / (Z_t[1:] + Z_t[:-1])

import def_models
wvlt_cfreq  = 32.0
wvlt_phase  = 0.0
wvlt_length = 0.128
wvlt_t, wvlt_amp = def_models.ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)
synth = np.convolve(wvlt_amp, RC_t, mode='same')
RC_t [ np.isnan(RC_t) ] = 0.0
synth [ np.isnan(synth) ] = 0.0

#%%

row1 = 0
row2 = 750

fig, t_ax = plt.subplots(ncols=3, nrows=2, constrained_layout=True, figsize = (6.75, 3.0))
vlim = max( np.abs(np.max(x_nuspan1[row1:row2, :])), np.abs(np.min(x_nuspan1[row1:row2, :])) )
high = np.max(x_nuspan1[row1:row2, :])
low = np.min(x_nuspan1[row1:row2, :])

for ax in t_ax.reshape(-1):
    ax.tick_params(width=0.5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)

vlim = 7000

props1 = dict(boxstyle='round', facecolor='white', alpha=0.2, edgecolor='None')

t_ax[0,0].imshow( np.fliplr(y[row1:row2, :]), cmap='seismic', vmin=-vlim, vmax=vlim, aspect='auto', alpha=0.6, extent=[1150,1350,3,0] )
t_ax[0,0].set_xticks( [1150, 1200, 1250, 1300, 1350] )
t_ax[0,0].set_ylabel('TIME (s)')
t_ax[0,0].set_xlabel('INLINE')
textstr = 'SEISMIC DATA'
t_ax[0,0].text(0.025, 0.95, textstr, transform=t_ax[0,0].transAxes, verticalalignment='top', bbox=props1)
rect = patches.Rectangle( (1300,0.9), 40, 0.4, linewidth=0.5, edgecolor='k', facecolor='None' )
t_ax[0,0].add_patch(rect)
textstr = '(a)'
t_ax[0,0].text(0.46, -0.38, textstr, transform=t_ax[0,0].transAxes, verticalalignment='top', bbox=props1, **csfont)

axe = t_ax[0,0]
bottom = axe.get_position().get_points()[0][1]
top =  axe.get_position().get_points()[1][1]
axf = axe.figure.add_axes([0.2515, bottom+0.146, 0.05, top-bottom-0.0675])
gain_synth = 1
axf.plot(gain_synth * synth, t[:-1], 'k', alpha = 0.9, linewidth=0.2)
axf.fill_betweenx(t[:-1], gain_synth * synth,  0, 
                gain_synth * synth > 0.0,
                color = 'k', alpha = 0.7, lw=0)
axf.set_ylim(0, 3)
axf.set_xlim(-0.5 ,0.5)
axf.invert_yaxis()
axf.set_yticklabels('')
axf.set_axis_off()
axf.grid()

row1 = 225
row2 = 325

col1 = 0
col2 = 40

im1 = t_ax[0,1].imshow( x_bpi[row1:row2, col1:col2], cmap='seismic', vmin=-vlim, vmax=vlim, alpha=0.8, extent=[1300,1340,row2*4/1000,row1*4/1000], aspect='auto' )
t_ax[0,1].set_ylabel('TIME (s)')
t_ax[0,1].set_xlabel('INLINE')
textstr = 'BPI'
props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='None')
t_ax[0,1].text(0.025, 0.95, textstr, transform=t_ax[0,1].transAxes, verticalalignment='top', bbox=props)
textstr = '(b)'
t_ax[0,1].text(0.46, -0.38, textstr, transform=t_ax[0,1].transAxes, verticalalignment='top', bbox=props1, **csfont)
axr_bpi = axe.figure.add_axes([0.495, bottom+0.146, 0.05, top-bottom-0.0675])
gain_synth = 1
axr_bpi.plot(gain_synth * RC_t, t[:-1], 'k', alpha = 0.9, linewidth=0.2)
axr_bpi.fill_betweenx(t[:-1], gain_synth * RC_t,  0, 
                gain_synth * RC_t > 0.0,
                color = 'k', alpha = 0.9, lw=0)
axr_bpi.set_ylim(row1*4/1000, row2*4/1000)
axr_bpi.set_xlim(-0.5 ,0.5)
axr_bpi.invert_yaxis()
axr_bpi.set_yticklabels('')
axr_bpi.set_axis_off()
axr_bpi.grid()


t_ax[0,2].imshow( x_fista[row1:row2, col1:col2], cmap='seismic', vmin=-vlim, vmax=vlim, alpha=0.8, extent=[1300,1340,row2*4/1000,row1*4/1000], aspect='auto' )
t_ax[0,2].set_yticks( () )
t_ax[0,2].set_xlabel('INLINE')
textstr = 'FISTA'
t_ax[0,2].text(0.025, 0.95, textstr, transform=t_ax[0,2].transAxes, verticalalignment='top', bbox=props)
textstr = '(c)'
t_ax[0,2].text(0.46, -0.38, textstr, transform=t_ax[0,2].transAxes, verticalalignment='top', bbox=props1, **csfont)
axr_fista = axe.figure.add_axes([0.824, bottom+0.146, 0.05, top-bottom-0.0675])
gain_synth = 1
axr_fista.plot(gain_synth * RC_t, t[:-1], 'k', alpha = 0.9, linewidth=0.2)
axr_fista.fill_betweenx(t[:-1], gain_synth * RC_t,  0, 
                gain_synth * RC_t > 0.0,
                color = 'k', alpha = 0.9, lw=0)
axr_fista.set_ylim(row1*4/1000, row2*4/1000)
axr_fista.set_xlim(-0.5 ,0.5)
axr_fista.invert_yaxis()
axr_fista.set_yticklabels('')
axr_fista.set_axis_off()
axr_fista.grid()



t_ax[1,0].imshow( x_sbl[row1:row2, col1:col2], cmap='seismic', vmin=-vlim, vmax=vlim, alpha=0.8, extent=[1300,1340,row2*4/1000,row1*4/1000], aspect='auto' )
t_ax[1,0].set_ylabel('TIME (s)')
t_ax[1,0].set_xlabel('INLINE')
textstr = 'SBL-EM'
t_ax[1,0].text(0.025, 0.95, textstr, transform=t_ax[1,0].transAxes, verticalalignment='top', bbox=props)
textstr = '(d)'
t_ax[1,0].text(0.46, -0.38, textstr, transform=t_ax[1,0].transAxes, verticalalignment='top', bbox=props1, **csfont)
axr_sbl = axe.figure.add_axes([0.165, bottom-0.34, 0.05, top-bottom-0.0675])
gain_synth = 1
axr_sbl.plot(gain_synth * RC_t, t[:-1], 'k', alpha = 0.9, linewidth=0.2)
axr_sbl.fill_betweenx(t[:-1], gain_synth * RC_t,  0, 
                gain_synth * RC_t > 0.0,
                color = 'k', alpha = 0.9, lw=0)
axr_sbl.set_ylim(row1*4/1000, row2*4/1000)
axr_sbl.set_xlim(-0.5 ,0.5)
axr_sbl.invert_yaxis()
axr_sbl.set_yticklabels('')
axr_sbl.set_axis_off()
axr_sbl.grid()


t_ax[1,1].imshow( x_nuspan1[row1:row2,col1:col2], cmap='seismic', vmin=-vlim, vmax=vlim, alpha=0.8, extent=[1300,1340,row2*4/1000,row1*4/1000], aspect='auto' )
t_ax[1,1].set_yticks(())
t_ax[1,1].set_xlabel('INLINE')
textstr = '$\it{Nu}$SPAN-$1$'
t_ax[1,1].text(0.025, 0.95, textstr, transform=t_ax[1,1].transAxes, verticalalignment='top', bbox=props)
textstr = '(e)'
t_ax[1,1].text(0.46, -0.38, textstr, transform=t_ax[1,1].transAxes, verticalalignment='top', bbox=props1, **csfont)
axr_nuspan1 = axe.figure.add_axes([0.495, bottom-0.34, 0.05, top-bottom-0.0675])
gain_synth = 1
axr_nuspan1.plot(gain_synth * RC_t, t[:-1], 'k', alpha = 0.9, linewidth=0.2)
axr_nuspan1.fill_betweenx(t[:-1], gain_synth * RC_t,  0, 
                gain_synth * RC_t > 0.0,
                color = 'k', alpha = 0.9, lw=0)
axr_nuspan1.set_ylim(row1*4/1000, row2*4/1000)
axr_nuspan1.set_xlim(-0.5 ,0.5)
axr_nuspan1.invert_yaxis()
axr_nuspan1.set_yticklabels('')
axr_nuspan1.set_axis_off()
axr_nuspan1.grid()


t_ax[1,2].imshow( x_nuspan2[row1:row2,col1:col2], cmap='seismic', vmin=-vlim, vmax=vlim, alpha=0.8, extent=[1300,1340,row2*4/1000,row1*4/1000], aspect='auto' )
t_ax[1,2].set_yticks(())
t_ax[1,2].set_xlabel('INLINE')
textstr = '$\it{Nu}$SPAN-$2$'
t_ax[1,2].text(0.025, 0.95, textstr, transform=t_ax[1,2].transAxes, verticalalignment='top', bbox=props)
textstr = '(f)'
t_ax[1,2].text(0.46, -0.38, textstr, transform=t_ax[1,2].transAxes, verticalalignment='top', bbox=props1, **csfont)
axr_nuspan2 = axe.figure.add_axes([0.824, bottom-0.34, 0.05, top-bottom-0.0675])
gain_synth = 1
axr_nuspan2.plot(gain_synth * RC_t, t[:-1], 'k', alpha = 0.9, linewidth=0.2)
axr_nuspan2.fill_betweenx(t[:-1], gain_synth * RC_t,  0, 
                gain_synth * RC_t > 0.0,
                color = 'k', alpha = 0.9, lw=0)
axr_nuspan2.set_ylim(row1*4/1000, row2*4/1000)
axr_nuspan2.set_xlim(-0.5 ,0.5)
axr_nuspan2.invert_yaxis()
axr_nuspan2.set_yticklabels('')
axr_nuspan2.set_axis_off()
axr_nuspan2.grid()

plt.tight_layout()
plt.savefig('fig_real_xline_{}.pdf'.format(xline), bbox_inches='tight', pad_inches = 0.01, dpi=600)
plt.close('all')