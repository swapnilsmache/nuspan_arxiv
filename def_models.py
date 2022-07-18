#################################
###### 1D Synthetic Traces ######
#################################
# Generate the seismic trace (y), dictionary/kernel matrix (D), reflecitvity function (x), and the time axis

def trace(wvlt_type, wvlt_cfreq, num_samples, num_traces, k):

    ####################
    # MODEL PARAMETERS #
    ####################
    import numpy as np
    import scipy

    # np.random.seed(4)       # create SAME random array every time

    #   Wavelet Parameters
    # wvlt_type = 'ricker'    # Valid values: 'ricker' or 'bandpass'
    wvlt_length= 128/1000   # Wavelet length in seconds
    wvlt_phase = 0.0        # Wavelet phase in degrees
    wvlt_scalar = 1.0       # Multiplier to scale wavelet amplitude (default = 1.0)
    # wvlt_cfreq = 30.0       # Ricker wavelet central frequency
    f2 = wvlt_cfreq - 15.0  # Bandpass wavelet low cut frequency
    f3 = wvlt_cfreq + 20.0  # Bandpass wavelet high cut frequency
    f1 = f2 - 5.0           # Bandpass wavelet low truncation frequency
    f4 = f3 + 15.0          # Bandpass wavelet high truncation frequency
    dt = 1/1000             # changing this from 0.0001 can affect the display quality

    dt_rc = dt
    t_max = dt_rc*(num_samples-1)

    # Generate wavelet
    if wvlt_type == 'ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)

    elif wvlt_type == 'bandpass':
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)

    # Apply amplitude scale factor to wavelet (to match seismic amplitude values)
    wvlt_amp = wvlt_scalar * wvlt_amp

    # Generate reflectivity function
    ref_range = np.round(np.linspace(-1, 1, 11), 1)          # reflectivity values in range [-1.0, 1.0], step = 0.2, randomly spaced in a (num_samples,1) vector
    time_axis = np.linspace(0, t_max*1000, round(t_max/dt_rc) + 1 )
    ref_model = []
    seismic_model = []
    padding = 100               # to avoid end-effects after convolving with wavelet
    N = num_samples - padding
    spikes = int(k*N)
    gap = 1

    from scipy.sparse import random

    for i in range(num_traces):
        # ref_locations = sorted(np.random.choice(N-((gap-1)*spikes-1), spikes, replace=False)) + (gap-1)*np.arange(spikes)
        ref_locations = sorted(np.random.choice(N-1, spikes)) + (gap-1)*np.arange(spikes)
        ref_values = np.random.choice(ref_range, int((N)*k))      # choose values of reflectivity from ref_range based on sparsity factor k
        ref_trace = np.zeros([N,])
        ref_trace[np.r_[ref_locations]] = ref_values[:]
        pad_zeros = int(padding/2)*[0]
        ref_trace = np.hstack([pad_zeros, ref_trace, pad_zeros])
        ref_model.append(ref_trace)
        ## Convolve reflectivity with wavelet to generate seismic trace
        seismic_trace = np.convolve(ref_trace, wvlt_amp, mode='same')
        seismic_model.append(seismic_trace)

    ref_model = np.array(ref_model, dtype=np.float32)
    seismic_model = np.array(seismic_model, dtype=np.float32)

    ## Generate wavelet convolution matrix
    x_trace = ref_model[0, :].copy()
    D_trace = convolution_matrix(wvlt_amp, len(x_trace), mode='same')

    # y = []
    # for i in range(num_traces):
    #     y_trace = np.dot(D_trace, ref_model[i, :])
    #     y.append(y_trace)
    # y = np.array(y)
    y = seismic_model.copy()
    x = ref_model.copy()

    return y.T, D_trace, x.T, time_axis, wvlt_t, wvlt_amp

def trace_mm(wvlt_type, wvlt_cfreq, num_samples, num_traces, k):

    ####################
    # MODEL PARAMETERS #
    ####################
    import numpy as np
    import scipy

    # np.random.seed(4)       # create SAME random array every time

    #   Wavelet Parameters
    # wvlt_type = 'ricker'    # Valid values: 'ricker' or 'bandpass'
    wvlt_length= 128/1000   # Wavelet length in seconds
    wvlt_phase = 0.0        # Wavelet phase in degrees
    wvlt_scalar = 1.0       # Multiplier to scale wavelet amplitude (default = 1.0)
    # wvlt_cfreq = 30.0       # Ricker wavelet central frequency
    f2 = wvlt_cfreq - 15.0  # Bandpass wavelet low cut frequency
    f3 = wvlt_cfreq + 20.0  # Bandpass wavelet high cut frequency
    f1 = f2 - 5.0           # Bandpass wavelet low truncation frequency
    f4 = f3 + 15.0          # Bandpass wavelet high truncation frequency
    dt = 2/1000             # changing this from 0.0001 can affect the display quality

    dt_rc = dt
    t_max = dt_rc*(num_samples-1)

    # Generate wavelet
    if wvlt_type == 'ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)

    elif wvlt_type == 'bandpass':
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)

    # Apply amplitude scale factor to wavelet (to match seismic amplitude values)
    wvlt_amp = wvlt_scalar * wvlt_amp

    # Generate reflectivity function
    ref_range = np.round(np.linspace(-1, 1, 41), 2)          # reflectivity values in range [-1.0, 1.0], step = 0.05, randomly spaced in a (num_samples,1) vector
    # print(ref_range)
    time_axis = np.linspace(0, t_max*1000, round(t_max/dt_rc) + 1 )
    ref_model = []
    seismic_model = []
    padding = 100               # to avoid end-effects after convolving with wavelet
    N = num_samples - padding
    spikes = int(k*N)
    gap = 1

    from scipy.sparse import random

    for i in range(num_traces):
        # ref_locations = sorted(np.random.choice(N-((gap-1)*spikes-1), spikes, replace=False)) + (gap-1)*np.arange(spikes)
        ref_locations = sorted(np.random.choice(N-1, spikes)) + (gap-1)*np.arange(spikes)
        ref_values = np.random.choice(ref_range, int((N)*k))      # choose values of reflectivity from ref_range based on sparsity factor k
        ref_trace = np.zeros([N,])
        ref_trace[np.r_[ref_locations]] = ref_values[:]
        pad_zeros = int(padding/2)*[0]
        ref_trace = np.hstack([pad_zeros, ref_trace, pad_zeros])
        ref_model.append(ref_trace)
        ## Convolve reflectivity with wavelet to generate seismic trace
        seismic_trace = np.convolve(ref_trace, wvlt_amp, mode='same')
        seismic_model.append(seismic_trace)

    ref_model = np.array(ref_model, dtype=np.float32)
    seismic_model = np.array(seismic_model, dtype=np.float32)

    ## Generate wavelet convolution matrix
    x_trace = ref_model[0, :].copy()
    D_trace = convolution_matrix(wvlt_amp, len(x_trace), mode='same')

    y = seismic_model.copy()
    x = ref_model.copy()

    return y.T, D_trace, x.T, time_axis, wvlt_t, wvlt_amp

def trace_real(wvlt_type, wvlt_cfreq, num_samples, num_traces, k):

    ####################
    # MODEL PARAMETERS #
    ####################
    import numpy as np
    import scipy

    # np.random.seed(4)       # create SAME random array every time

    #   Wavelet Parameters
    # wvlt_type = 'ricker'    # Valid values: 'ricker' or 'bandpass'
    wvlt_length= 128/1000   # Wavelet length in seconds
    wvlt_phase = 0.0        # Wavelet phase in degrees
    wvlt_scalar = 1.0       # Multiplier to scale wavelet amplitude (default = 1.0)
    # wvlt_cfreq = 30.0       # Ricker wavelet central frequency
    f2 = wvlt_cfreq - 15.0  # Bandpass wavelet low cut frequency
    f3 = wvlt_cfreq + 20.0  # Bandpass wavelet high cut frequency
    f1 = f2 - 5.0           # Bandpass wavelet low truncation frequency
    f4 = f3 + 15.0          # Bandpass wavelet high truncation frequency
    dt = 4/1000             # changing this from 0.0001 can affect the display quality

    dt_rc = dt
    t_max = dt_rc*(num_samples-1)

    # Generate wavelet
    if wvlt_type == 'ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)

    elif wvlt_type == 'bandpass':
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)

    # Apply amplitude scale factor to wavelet (to match seismic amplitude values)
    wvlt_amp = wvlt_scalar * wvlt_amp

    # Generate reflectivity function
    ref_range = np.round(np.linspace(-1, 1, 41), 2)          # reflectivity values in range [-1.0, 1.0], step = 0.05, randomly spaced in a (num_samples,1) vector
    # print(ref_range)
    time_axis = np.linspace(0, t_max*1000, round(t_max/dt_rc) + 1 )
    ref_model = []
    seismic_model = []
    padding = 100               # to avoid end-effects after convolving with wavelet
    N = num_samples - padding
    spikes = int(k*N)
    gap = 1

    from scipy.sparse import random

    for i in range(num_traces):
        # ref_locations = sorted(np.random.choice(N-((gap-1)*spikes-1), spikes, replace=False)) + (gap-1)*np.arange(spikes)
        ref_locations = sorted(np.random.choice(N-1, spikes)) + (gap-1)*np.arange(spikes)
        ref_values = np.random.choice(ref_range, int((N)*k))      # choose values of reflectivity from ref_range based on sparsity factor k
        ref_trace = np.zeros([N,])
        ref_trace[np.r_[ref_locations]] = ref_values[:]
        pad_zeros = int(padding/2)*[0]
        ref_trace = np.hstack([pad_zeros, ref_trace, pad_zeros])
        ref_model.append(ref_trace)
        ## Convolve reflectivity with wavelet to generate seismic trace
        seismic_trace = np.convolve(ref_trace, wvlt_amp, mode='same')
        seismic_model.append(seismic_trace)

    ref_model = np.array(ref_model, dtype=np.float32)
    seismic_model = np.array(seismic_model, dtype=np.float32)

    ## Generate wavelet convolution matrix
    x_trace = ref_model[0, :].copy()
    D_trace = convolution_matrix(wvlt_amp, len(x_trace), mode='same')

    y = seismic_model.copy()
    x = ref_model.copy()

    return y.T, D_trace, x.T, time_axis, wvlt_t, wvlt_amp

### Generate noisy signal (y_noisy)

def trace_noisy(y, target_snr_db):
    import numpy as np
    noiseAmplitude = 1
    noiseSigma = 1
    y_size = y.shape[0]

    noise = noiseAmplitude * np.random.normal(0, noiseSigma, y_size)

    cleanSound = y.copy()
    y_noisy = []
    signal_power = np.sum(np.abs(cleanSound**2))/y_size
    noise_power = np.sum(np.abs(noise**2))/y_size
    snr_initial = 10* np.log10(signal_power/noise_power)

    k = (signal_power/noise_power)*10**(-target_snr_db/10)
    new_noise = np.sqrt(k)*noise

    new_noise_power = np.sum(np.abs(new_noise**2))/y_size
    snr_new = 10 * np.log10(signal_power/new_noise_power)

    y_noisy = cleanSound + new_noise
    # noisy_signal = cleanSound + new_noise
    # y_noisy = np.array(noisy_signal)

    return y_noisy



########################
# FUNCTION DEFINITIONS #
########################

# Modified from original code by Wes Hamlyn (https://github.com/seg/tutorials-2014/blob/master/1412_Tuning_and_AVO/tuning_prestack.py)
# Reference: Hamlyn, W. (2014). Thin beds, tuning, and AVO. The Leading Edge, 33(12), 1394-1396. https://doi.org/10.1190/tle33121394.1

def ricker(cfreq, phase, dt, wvlt_length):
    '''
    Generate a ricker (Mexican hat) wavelet
    
    Usage:
    ------
    t, wvlt = wvlt_ricker(cfreq, phase, dt, wvlt_length)
    
    cfreq: central frequency of wavelet in Hz
    phase: wavelet phase in degrees
    dt: sample rate in seconds
    wvlt_length: length of wavelet in seconds
    '''
    
    import numpy as np
    import scipy.signal as signal
    
    nsamp = int(wvlt_length/dt + 1)
    t_max = wvlt_length*0.5
    t_min = -t_max
    
    t = np.arange(t_min, t_max, dt)
    
    t = np.linspace(-wvlt_length/2, (wvlt_length-dt)/2, int(wvlt_length/dt))
    wvlt = (1.0 - 2.0*(np.pi**2)*(cfreq**2)*(t**2)) * np.exp(-(np.pi**2)*(cfreq**2)*(t**2))     # generate the ricker (mexican hat) wavelet
    
    if phase != 0:
        phase = phase*np.pi/180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase)*wvlt - np.sin(phase)*wvlth
    
    return t, wvlt

def wvlt_bpass(f1, f2, f3, f4, phase, dt, wvlt_length):
    '''
    Calculate a trapezoidal bandpass wavelet
    
    Usage:
    ------
    t, wvlt = wvlt_ricker(f1, f2, f3, f4, phase, dt, wvlt_length)
    
    f1: Low truncation frequency of wavelet in Hz
    f2: Low cut frequency of wavelet in Hz
    f3: High cut frequency of wavelet in Hz
    f4: High truncation frequency of wavelet in Hz
    phase: wavelet phase in degrees
    dt: sample rate in seconds
    wvlt_length: length of wavelet in seconds
    '''
    
    from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
    
    nsamp = int(wvlt_length/dt + 1)
        
    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    aspec = freq*0.0
    pspec = freq*0.0
    
    # Calculate slope and y-int for low frequency ramp
    M1 = 1/(f2-f1)
    b1 = -M1*f1
    
    # Calculate slop and y-int for high frequency ramp
    M2 = -1/(f4-f3)
    b2 = -M2*f4
    
    # Build initial frequency and filter arrays
    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    filt = np.zeros(nsamp)
    
    # Build LF ramp
    idx = np.nonzero((np.abs(freq)>=f1) & (np.abs(freq)<f2))
    filt[idx] = M1*np.abs(freq)[idx]+b1
    
    # Build central filter flat
    idx = np.nonzero((np.abs(freq)>=f2) & (np.abs(freq)<=f3))
    filt[idx] = 1.0
    
    # Build HF ramp
    idx = np.nonzero((np.abs(freq)>f3) & (np.abs(freq)<=f4))
    filt[idx] = M2*np.abs(freq)[idx]+b2
    
    # Unshift the frequencies and convert filter to fourier coefficients
    filt2 = ifftshift(filt)
    Af = filt2*np.exp(np.zeros(filt2.shape)*1j)
    
    # Convert filter to time-domain wavelet
    wvlt = fftshift(ifft(Af))
    wvlt = np.real(wvlt)
    wvlt = wvlt/np.max(np.abs(wvlt))                    # normalize wavelet by peak amplitude
    
    # Generate array of wavelet times
    t = np.linspace(-wvlt_length*0.5, wvlt_length*0.5, nsamp)
        
    # Apply phase rotation if desired
    if phase != 0:
        phase = phase*np.pi/180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase)*wvlt - np.sin(phase)*wvlth
    
    return t, wvlt

def calc_rc(vp_model, rho_model):                       # calculate reflectivity coefficients from impedance(=rho*vp_model)
    '''
    ref_coeffs = calc_rc(vp_model, rho_model)
    '''
    
    nlayers = len(vp_model)
    nint = nlayers - 1
    
    ref_coeffs = []
    for i in range(0, nint):
        buf1 = vp_model[i+1]*rho_model[i+1]-vp_model[i]*rho_model[i]
        buf2 = vp_model[i+1]*rho_model[i+1]+vp_model[i]*rho_model[i]
        buf3 = buf1/buf2
        ref_coeffs.append(buf3)
    
    return ref_coeffs

def calc_times(z_int, vp_model):                        # calculate time taken to travel interface with depth z_int and p-wave velocity vp_model
    '''
    t_int = calc_times(z_int, vp_model)
    '''
    
    nlayers = len(vp_model)
    nint = nlayers - 1

    t_int = []
    for i in range(0, nint):
        if i == 0:
            tbuf = z_int[i]/vp_model[i]                 # first reflection. one-way travel time
            t_int.append(tbuf)
        else:
            zdiff = z_int[i]-z_int[i-1]                 # thickness of layer
            tbuf = 2*zdiff/vp_model[i] + t_int[i-1]     # two-way travel time for the layer with thickness zdiff, added to previous calculated time. time is absolute time required for reflection to be detected/recorded at ground surface
            t_int.append(tbuf)
    
    return t_int

def digitize_model(ref_coeffs, t_int, t):
    '''
    rc = digitize_model(rc, t_int, t)
    
    rc = reflection coefficients corresponding to interface times
    t_int = interface times
    t = regularly sampled time series defining model sampling
    '''
    
    import numpy as np
    
    nlayers = len(ref_coeffs)
    nint = nlayers - 1
    nsamp = len(t)
    
    rc = list(np.zeros(nsamp,dtype='float'))            # vector of zeros. size of one trace
    lyr = 0
    
    for i in range(0, nsamp):

        if t[i] >= t_int[lyr]:                          # condition first satisfied when i reaches value such that t[i] = 0.2, that is, first horizontal interface.
            rc[i] = ref_coeffs[lyr]                     # rc is assigned a value only when {above}. otherwise all zeros
            lyr = lyr + 1                               # lyr toggles between 0 and 1. assigns corresponding ref_coeffs values to rc at those points

        if lyr > nint:
            break
            
    return rc

#########################
###### Wedge Model ######
#########################

def wedge(wvlt_type, wvlt_cfreq,model_type):
    #   3-Layer Model Parameters [Layer1, Layer2, Layer 3]
    vel0 = 2000.0
    rho0 = 1.0
    if model_type == 'nn':    
        vp_model = [vel0, vel0, vel0]                   # P-wave velocity (m/s)
        rho_model = [rho0, rho0/3.0, rho0/(3.0*3.0)]    # Density (g/cc)
    elif model_type == 'pp':
        vp_model = [vel0, vel0, vel0]                   # P-wave velocity (m/s)
        rho_model = [rho0, rho0*3.0, rho0*3.0*3.0]      # Density (g/cc)
    elif model_type == 'np':
        vp_model = [vel0, vel0, vel0]                   # P-wave velocity (m/s)
        rho_model = [rho0, rho0/3.0, rho0]              # Density (g/cc)
    elif model_type == 'pn':
        vp_model = [vel0, vel0, vel0]                   # P-wave velocity (m/s)
        rho_model = [rho0, rho0*3.0, rho0]              # Density (g/cc)

    dz_min = 0.0                                        # Minimum thickness of Layer 2 (m)
    dz_max = 50.0                                       # Maximum thickness of Layer 2 (m)
    dz_step= 2.0                                        # Thickness step from trace-to-trace (normally 1.0 m)

    #   Wavelet Parameters
    wvlt_length= 0.128                                  # Wavelet length in seconds
    wvlt_phase = 0.0                                    # Wavelet phase in degrees
    wvlt_scalar = 1.0                                   # Multiplier to scale wavelet amplitude (default = 1.0)
    f2 = wvlt_cfreq - 20.0                              # Bandpass wavelet low cut frequency
    f3 = wvlt_cfreq + 20.0                              # Bandpass wavelet high cut frequency
    f1 = f2 - 5.0                                       # Bandpass wavelet low truncation frequency
    f4 = f3 + 15.0                                      # Bandpass wavelet high truncation frequency

    #   Trace Parameters
    tmin = 0.00
    tmax = 0.299
    dt = 1/1000                                         # changing this from 0.0001 can affect the display quality

    ################
    # COMPUTATIONS #
    ################

    #   Some handy constants
    nlayers = len(vp_model)                             # number of layers
    nint = nlayers - 1                                  # number of interfaces
    nmodel = int((dz_max-dz_min)/dz_step+1)             # number of traces

    # Generate wavelet
    if wvlt_type == 'ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)

    elif wvlt_type == 'bandpass':
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)

    # Apply amplitude scale factor to wavelet (to match seismic amplitude values)
    wvlt_amp = wvlt_scalar * wvlt_amp

    #   Calculate reflectivities from model parameters
    ref_coeffs = calc_rc(vp_model, rho_model)
    # ref_coeffs = [-1, 1]

    import numpy as np
    y = []
    x = []
    lyr_times = []
    for model in range(0, nmodel):     # for trace in range(0, number of traces)
        
        #   Calculate interface depths
        z_int = [100.0]                                 # starting depth, depth of top horizontal interface
        z_int.append(z_int[0]+dz_min+dz_step*model)     # endpoints of wedge for that particular wedge [500.0, depth to dipping interface]
        
        #   Calculate interface times
        t_int = calc_times(z_int, vp_model)             # travel-time for current trace
        lyr_times.append(t_int)                         # travel-times for all traces
        
        #   Digitize 3-layer model
        nsamp = int((tmax-tmin)/dt) + 1
        t = []
        for i in range(0,nsamp):
            t.append(i*dt)
            
        rc = digitize_model(ref_coeffs, t_int, t)
        x.append(rc)
        
        #   Convolve wavelet with reflectivities (to create synthetic seismic trace)
        syn_buf = np.convolve(rc, wvlt_amp, mode='same')
        syn_buf = list(syn_buf)
        y.append(syn_buf)
        # print "finished step %i" % (model)
        # print ("finished step ", model)

    # print(ref_coeffs)

    x = np.array(x)
    y = np.array(y)
    t = np.array(t)*1000
    lyr_times = np.array(lyr_times)
    lyr_indx = np.array(np.round(lyr_times/dt), dtype='int16')  # indices where seismic trace has non-zero value, that is, at the two interfaces
    # print(lyr_times)

    # Use the transpose because rows are traces;
    # columns are time samples.
    tuning_trace = np.argmax(np.abs(y.T)) % y.T.shape[1]        # trace for which value in s(t) is highest
    tuning_thickness = tuning_trace * dz_step

    # Generate wavelet convolution matrix
    [num_traces, num_samples] = y.shape                         # [number of traces, number of samples in each trace]
    D_trace = convolution_matrix(wvlt_amp, len(x[0,:]), mode='same')

    return y, D_trace, x, t, wvlt_t, wvlt_amp, lyr_times, lyr_indx



# Author: Jake VanderPlas
# LICENSE: MIT
# https://gist.github.com/jakevdp/d2d453d987ccb92f55ff574818cced33#file-convolution_matrix-py

def convolution_matrix(x, N=None, mode='same'):
    """Compute the Convolution Matrix

    This function computes a convolution matrix that encodes
    the computation equivalent to ``numpy.convolve(x, y, mode)``

    Parameters
    ----------
    x : array_like
        One-dimensional input array
    N : integer (optional)
        Size of the array to be convolved. Default is len(x).
    mode : {'full', 'valid', 'same'}, optional
        The type of convolution to perform. Default is 'full'.
        See ``np.convolve`` documentation for details.

    Returns
    -------
    C : ndarray
        Matrix operator encoding the convolution. The matrix is of shape
        [Nout x N], where Nout depends on ``mode`` and the size of ``x``. 

    Example
    -------
    >>> x = np.random.rand(10)
    >>> y = np.random.rand(20)
    >>> xy = np.convolve(x, y, mode='full')
    >>> C = convolution_matrix(x, len(y), mode='full')
    >>> np.allclose(xy, np.dot(C, y))
    True

    See Also
    --------
    numpy.convolve : direct convolution operation
    scipy.signal.fftconvolve : direct convolution via the
                               fast Fourier transform
    scipy.linalg.toeplitz : construct the Toeplitz matrix
    """
    import numpy as np

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x should be 1-dimensional")

    M = len(x)
    N = M if N is None else N

    if mode == 'full':
        Nout = M + N - 1
        offset = 0
    elif mode == 'valid':
        Nout = max(M, N) - min(M, N) + 1
        offset = min(M, N) - 1
    elif mode == 'same':
        Nout = max(N, M)
        offset = (min(N, M) - 1) // 2
    else:
        raise ValueError("mode='{0}' not recognized".format(mode))

    xpad = np.hstack([x, np.zeros(Nout)])
    n = np.arange(Nout)[:, np.newaxis]
    m = np.arange(N)
    return xpad[n - m + offset]
