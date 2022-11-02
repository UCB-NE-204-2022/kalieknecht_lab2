import numpy as np
import h5py

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



def normalize_minmax(metric):
    '''
    Minmax normalize metric of interest

    Minmax normalization computed manually

    x' = (x - min(x)) / (max(x) - min(x))
    
    Parameters
    ----------
    metric: 1D array of num
        metric of interest
    
    Returns
    -------
    scaled_metric: 1D array of num
        minmax normalized metric of interest
    '''
    scaled_metric = (metric - metric.min()) / (metric.max() - metric.min())
    return scaled_metric

def normalize_standard(metric):
    '''
    Standard normalize metric of interest

    Standard normalization computed manually

    x' = (x - mean(x)) / std(x)
    
    Parameters
    ----------
    metric: 1D array of num
        metric of interest
    
    Returns
    -------
    scaled_metric: 1D array of num
        standard normalized metric of interest
    '''
    scaled_metric = (metric - metric.mean()) / metric.std()
    return scaled_metric
