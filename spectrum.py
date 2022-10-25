import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_prominences

def find_trapezoid_heights(filtered_waveforms):
    '''
    Find trapezoid heights for all waveforms
    
    Parameters
    ----------
    filtered_waveforms: np.array
        array of filtered waveforms
        
    Returns
    -------
    trapezoidal_heights: np.array
        1d array of trapezoid height for each waveform
    '''
    return filtered_waveforms.max(axis=1)

def plot_trapezoid_height_histogram(trapezoid_heights,
        bins=8000,
        xlims=[0,1e17],
        semilogy=False,
        save_name=None):
    '''
    Plot trapezoid height histogram
    
    Parameters
    ----------
    trapezoid_heights: np.array
        height of trapezoid in filtered waveforms
    bins: int
        number of bins to use (default 8000)
    xlims: list
        min and max x in plot
    semilogy: bool
        whether to use semilog y or not
    save_name: str
        if not empty saves plot
    '''
    plt.figure()
    plt.hist(trapezoid_heights,bins=bins)
    plt.xlim(xlims)
    if semilogy:
        plt.semilogy()
    plt.show()
    if save_name is not None:
        plt.savefig(save_name)
    return

def make_calibration_spectrum(trapezoid_heights,
    bins=5000,
    max_scaler=110):
    '''
    Make calibration spectrum
    
    Params
    ------
    max_scaler: float
        multiplication factor to trap height min to scale max of histogram
    
    Returns
    -------
    
    '''
    calib_spectrum = np.histogram(trapezoid_heights, bins=bins, range=(trapezoid_heights.min(),trapezoid_heights.min()*max_scaler))
    
def gaus(x, A, x0, sigma):
    '''
    Definition of gaussian for use in FWHM/peak fitting
    '''
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def find_peaks(calibration_spectrum,
        prominence=100):
    '''
    Find peaks in spectrum for calibration
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    peaks, prominences = find_peaks(calibration_spectrum[0], prominence = prominence)
