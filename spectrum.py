import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_prominences

class spectrum():
    '''
    Object to create and hold spectrum
    '''
    def __init__(self,
        filtered_waveforms,
        bins=5000,
        max_scaler=5,
        ):
        '''
        Initialize spectrum
        
        Parameters
        ----------
        filtered_waveforms: np.array
            array of filtered waveforms
        bins: int
            number of bins in spectrum
        max_scaler: float
            scaler value to max calibration histogram
        
        Returns
        --------
        spectrum: object
            histogrammed counts
        
        '''
        self.bins = bins
        
        # find trapezoid heights
        self.trapezoid_heights = find_trapezoid_heights(filtered_waveforms)
        
        self.make_calibration_spectrum(max_scaler=max_scaler)
        
    def run_full_pipeline(self,
        energies,
        smoothed=False,
        prominence=60):
        '''
        Runs full pipeline of spectrum tools
        
        Finds gamma_peaks
        
        Parameters
        ----------
        
        
        Returns
        -------
        '''
        # find gamma ray peaks and prominences
        self.find_gamma_peaks(prominence=prominence,smoothed=smoothed)
        
        # find energy calibration
        self.find_energy_calibration(energies)
        
        # find FWHMs
        self.find_energy_resolution()
        print('Done!')
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
        counts: ndarry
            counts in each histogrammed bin
        channels: ndarray
            bin_edges of histogram
        '''
        self.counts, self.channels = np.histogram(self.trapezoid_heights, bins=self.bins, range=(self.trapezoid_heights.min(),self.trapezoid_heights.min()*max_scaler))
        
    def smooth_spectrum(self,
        window_length=5,
        polyorder=1,
        show_plot=False,
        plot_savefile=None):
        '''
        Apply sagvol filter to smooth spectrum for peak fitting
        
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
        print('Smoothing spectrum with Savgol filter')
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
        smoothed=False,
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
            Whether to use smoothed version of spectrum for peak finding
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
        # find peaks on smoothed spectrum
        if smoothed:
            self.smooth_spectrum(window_length=smoothed_window_length)
            print('Finding Peaks')
            self.peaks, self.prominences = find_peaks(self.smoothed_counts, prominence = prominence)
        
        # find peaks on raw spectrum
        else:
            print('Finding Peaks')
            self.peaks, self.prominences = find_peaks(self.counts, prominence = prominence)
        
        # show plot, if desired
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
        #TODO rearrange so the peak fitting tolerance changes based on desired energy
        
        Parameters
        ----------
        energies: ndarray
            array of expected gamma ray energies (keV)
        
        Returns
        -------
        calibration_channels: ndarray
            array of channels associated with fitted peaks
        bin_energies: ndarray
            array of energies associated with calibration channels
        '''
        print('Finding energy calibration')
        # store energies to class
        self.energies = energies
        
        # check that number of energies matches number of peaks
        assert len(self.peaks) == len(self.energies), "Number of peaks ("+str(len(self.peaks))+") and energies ("+str(len(self.energies))+") do not match. Change prominence."
        
        # grab calibration channels for peaks
        self.calibration_channels = self.channels[1:][self.peaks]
        
        # find linear fit
        #TODO actually use a linear fit instead of grade school fitting 
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
            plot of spectrum with energy in keV on x-axis
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
            plot of Channel Energy in keV vs Channel
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
        fwhms: ndarray
            fwhm of fitted peaks in keV
        '''
        print('Finding energy calibration')
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
        plot_savefile: str
            plot savefile name, not saved if left empty
        
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
        1D array of trapezoid height for each waveform
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
