import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_prominences

class spectrum():
    '''
    Object to create spectrum
    '''
    def __init__(self,
        filtered_waveforms,
        bins=5000,
        ):
        '''
        Initialize spectrum
        
        Parameters
        ----------
        filtered_waveforms: np.array
            array of filtered waveforms
        bins: int
            number of bins in spectrum
        
        Returns
        --------
        Spectrum: object
            histogrammed counts
        
        '''
        self.bins = bins
        
        # find trapezoid heights
        self.trapezoid_heights = find_trapezoid_heights(filtered_waveforms)
        
        self.make_calibration_spectrum()
        
    def run_full_pipeline(self,
        energies,
        prominence=60):
        '''
        Run full pipeline
        '''
        # find gamma ray peaks and prominences
        self.find_gamma_peaks(prominence=prominence)
        
        # find energy calibration
        self.find_energy_calibration(energies)
        
        # find FWHMs
        self.find_energy_resolution()
        print(self.fwhms)
        
    def make_calibration_spectrum(self,
        max_scaler=5):
        '''
        Make calibration spectrum
        
        Params
        ------
        max_scaler: float
            multiplication factor to trap height min to scale max of histogram
        
        Returns
        -------
        
        '''
        self.counts, self.channels = np.histogram(self.trapezoid_heights, bins=self.bins, range=(self.trapezoid_heights.min(),self.trapezoid_heights.min()*max_scaler))
        
    def smooth_spectrum(self,
        window_length=5,
        polyorder=1,
        show_plot=False,
        plot_savefile=None):
        '''
        Apply sagvol filter to smooth spectrum to smooth for peak fitting
        
        See scipy.signal.savgol_filter for more information
        
        Parameters
        ----------
        window_length: int
            Length of filter window
        polyorder: int
            Order of polynomial to fit samples
        show_plot: bool
            Whether to show plot of saved spectrum
        plot_savefile: str
            plot savefile name, not saved if left empty
        
        Returns
        -------
        
        '''
        self.smoothed_counts = savgol_filter(self.counts, window_length, polyorder)
        if show_plot:
            plt.figure()
            plt.plot(self.channels[1:], self.counts,label='Spectrum')
            plt.plot(self.channels[1:], self.smoothed_counts,label='Savgol Smoothed Spectrum')
            plt.show()
            if plot_savefile is not None:
                plt.savefig(plot_savefile)
        
    def find_gamma_peaks(self,
        prominence=60,
        smoothed=True,
        smoothed_window_length=5,
        show_plot=False,
        plot_savefile=None):
        '''
        Find gamma-ray peaks
        
        See scipy.signal.find_peaks for more information
        
        Parameters 
        ----------
        prominence: float
            Required prominence of peaks
        smoothed: bool
            Whether to use smoothed version of spectrum
        show_plot: bool
            Whether to show plot of saved spectrum
        plot_savefile: str
            plot savefile name, not saved if left empty
           
        Returns
        -------
        peaks: np.array
            Indices of peaks
        prominences: np.array
            Prominences for each peak in peaks
        
        '''
        if smoothed:
            self.smooth_spectrum(window_length=smoothed_window_length)
            self.peaks, self.prominences = find_peaks(self.smoothed_counts, prominence = prominence)
        else:
            self.peaks, self.prominences = find_peaks(self.counts, prominence = prominence)
        
        if show_plot:
            plt.figure()
            plt.plot(self.channels[1:], self.counts)
            plt.plot(self.channels[1:][self.peaks], self.counts[self.peaks], 'x', markersize = 5)
            plt.xlabel('Channel Number')
            plt.ylabel('Counts')
            plt.show()
            if plot_savefile is not None:
                plt.savefig(plot_savefile)
                
    def find_energy_calibration(self,
        energies):
        '''
        Find linear energy calibration
        
        Parameters
        ----------
        energies: np.array
            array of expected gamma ray energies (keV)
        
        Returns
        -------
        energy_cal
        '''
        self.energies = energies
        assert len(self.peaks) == len(self.energies), "Number of peaks ("+str(len(self.peaks))+") and energies ("+str(len(self.energies))+") do not match. Change prominence."
        self.calibration_channels = self.channels[1:][self.peaks]
        self.slope = (energies[-1] - energies[0]) / (self.calibration_channels[-1] - self.calibration_channels[0])
        self.intercept = energies[0] - self.calibration_channels[0] * self.slope
        self.bin_energies = self.channels * self.slope + self.intercept
        
    def plot_energy_calibrated_spectrum(self,
        show_calibrated_peaks=True,
        semilogy=False,
        plot_savefile=None):
        '''
        Plot Energy calibrated spectrum
        
        Parameters
        ----------
        show_calibrated_peaks: bool
            whether to emphasized calibration peaks
        semilogy: bool
            whether to plot y on log axis
        plot_savefile: str
            plot savefile name, not saved if left empty
        
        Returns
        -------
        plot: matplotlib.pyplot.figure
        '''
        plt.figure()
        plt.plot(self.bin_energies[1:], self.counts)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        if show_calibrated_peaks:
            plt.plot(self.bin_energies[1:][self.peaks], self.counts[self.peaks], 'x', markersize = 5)
        if semilogy:
            plt.semilogy()
        plt.show()
        if plot_savefile is not None:
            plt.savefig(plot_savefile)
            
    def plot_energy_calibration(self,
        plot_savefile=None):
        '''
        Plot energy calibration and linear fit
        
        Parameters
        ----------
        plot_savefile: str
            plot savefile name, not saved if left empty
        
        Return
        ------
        plot: matplotlib.pyplot.figure
        '''
        plt.figure()
        plt.scatter(self.calibration_channels,self.energies,label='Calibration Peaks')
        plt.plot(self.channels,self.bin_energies,label='Linear Fit')
        plt.xlabel('Channel Number')
        plt.ylabel('Energy (keV)')
        plt.show()
        if plot_savefile is not None:
            plt.savefig(plot_savefile)
    
    def find_energy_resolution(self):
        '''
        Find Energy resolution of peaks
        
        Parameters
        ----------
        
        Returns
        -------
        fwhm: np.array
            fwhm of peaks in keV
        '''
        self.fwhms = np.zeros(len(self.peaks))
        i=0
        for peak in self.peaks:
            fwhm_high = next(idx for idx, val in zip(self.channels[peak:], self.counts[peak:]) if val <= 0.5 * self.counts[peak])
            
            fwhm_low = next(idx for idx, val in zip(reversed(self.channels[:peak]), reversed(self.counts[:peak])) if val <= 0.5 * self.counts[peak])
            
            fwhm = (fwhm_high * self.slope + self.intercept) - (fwhm_low * self.slope + self.intercept)
            self.fwhms[i] = fwhm
            i=i+1
    
    def plot_energy_resolution(self,
        plot_savefile=None):
        '''
        Plot energy resolution of fitted peaks
        
        Parameters
        ----------
        
        Returns
        -------
        plt: matplotlib.pyplot.figure
        
        '''
        plt.figure()
        plt.plot(self.energies,self.fwhms)
        plt.scatter(self.energies,self.fwhms)
        plt.title('Energy Resolution')
        plt.ylabel('FWHM (keV)')
        plt.xlabel('Energy (keV)')
        plt.show()
        if plot_savefile is not None:
            plt.savefig(plot_savefile)
        
        
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

def gaus(x, A, x0, sigma):
    '''
    Definition of gaussian for use in FWHM/peak fitting
    '''
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
