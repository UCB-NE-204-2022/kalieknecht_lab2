import numpy as np
import h5py
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def find_activity(t12,A0,time_elapsed):
    '''
    Find activity of source after time_elapsed since source born date
    
    Parameters
    ----------
    t12: float
        half life of source
    A0: float
        activity of source at born date
    time_elapsed: float
        time elapsed since source born date, in same units as t12
    
    Returns
    -------
    A: float
        activity in same units as A0 after time_elapsed
    '''
    decay_constant = np.log(2)/t12
    return A0 * np.exp(-decay_constant*time_elapsed)

def import_data(filename):
    '''
    Import and clean data. Removes duplicate events and converts to
    np.int16 type (instead of unsigned int)
    
    Parameters
    ----------
    filname: str
        h5 file path
    Returns
    -------
    raw_data: np.array
        array of raw waveforms, conveted to np.int16
    event_data: np.array
    '''
    # open file with h5py
    f_data = h5py.File(filename,'r')
    
    # extract raw and event data
    raw_data = np.int16(f_data['raw_data'][()])
    event_data = f_data['event_data'][()]
    
    # clean data (remove duplicate waveforms)
    raw_data, event_data = clean_raw_waveforms(raw_data,event_data)
    
    return raw_data, event_data

def clean_raw_waveforms(waveforms,events):
    '''
    Clean raw waveforms. Removes repeated data at the end of data acquisition
    
    Parameters
    ----------
    waveforms: np.array
        array of raw waveforms
    events: np.array
        array of event data
    
    Returns
    -------
    waveforms_cleaned: np.array
        cleaned waveforms
    events_cleaned: np.array
        cleaned events
    
    '''
    unique_waves, index = np.unique(events['timestamp'],return_index=True)
    events_cleaned = events[index]
    waveforms_cleaned = waveforms[index]
    return waveforms_cleaned, events_cleaned

def subtract_baseline(waveforms,
    baseline_end=100):
    '''
    Find ave baseline for each pulse and subtract from raw waveforms
    
    Parameters
    ----------
    baseline_end: int
        end integer of baseline data
    
    Returns
    -------
    bkg_subtracted_waveforms: np.array
        waveforms with baseline subtracted

    '''
    # find average baseline for each pulse
    ave_baseline = np.mean(waveforms[:,:baseline_end],axis=1)
    
    # subtract baseline from each waveform
    bkg_subtracted_waveforms = waveforms - ave_baseline[:,None]
    
    return bkg_subtracted_waveforms

def exponential(t, a, tau):
    return a * np.exp(-t / tau)

def fit_tau(waveform, 
        pre_sample_length=1100, 
        fit_length = 2000,
        show_plot=False,
        plot_save_name=None):
    '''
    Find time constant (tau) of waveform
    
    Parameters
    ----------
    waveform: np.array
        raw waveform data
    pre_sample_length: int
        start of decay fit
    fit_length: int
        length of data to fit
    show_plot: bool
        whether to show plot of waveform and exponential fit
    plot_save_name: str
        plot savename (include file extension)
    
    Returns
    -------
    tau: float
        fitted tau value for waveform
    '''
    
    # grab portion of waveform for fitting
    decay_waveform = waveform[pre_sample_length:pre_sample_length+fit_length]
    
    x = np.arange(0, fit_length)
    x_norm = (x - x[0]) / (x[-1] - x[0])
    
    # initial guesses for values
    tau_0 = 10000/fit_length
    a_0 = decay_waveform[0]
    popt, pcov = curve_fit(exponential, x_norm, decay_waveform, p0=(a_0, tau_0))
    a, tau_norm = popt
    tau = tau_norm * fit_length
    if show_plot:
        plt.figure()
        fit_vals = exponential(x_norm, a, tau_norm)
        plt.plot(waveform,label='Raw Waveform')
        plt.plot(x+pre_sample_length, fit_vals,label='Exponential Fit')
        plt.xlabel('Time (Clock Cycles)')
        plt.ylabel('Magnitued (ADC Units)')
        plt.text(3800,25,r'$\tau$='+str(round(tau,2)))
        plt.legend()
        plt.show()
        if plot_save_name is not None:
            plt.savefig(plot_save_name)
    return tau
