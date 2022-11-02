import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class JordanovFilter():
    '''
    We use the notation used in
    V.T. Jordanoo, G.F. Knoll/Nucl. Instr. and Meth . in Phys. Res. A 345 (1994) 337-345
    
    Code by Jaewon
    
    '''
    def __init__(self, 
        peaking_time, 
        gap_time, 
        decay_time):
        '''
        Initialize trapezoidal filter
        
        Parameters
        ----------
        peaking_time: float
            peaking time
        gap_time: float
            gap time
        decay_time: float
            decay time
        
        Returns
        -------
        TrapezoidalFilter: TrapezoidalFilter
            TrapezoidalFilter object
        '''
        self.peaking_time = peaking_time
        self.gap_time = gap_time
        self.decay_time = decay_time
        self.k = None
        self.l = None
        self.M = None
        
    def filter_waveform(self, 
        raw_data, 
        sampling_interval=4e-9, 
        normalize=True):
        '''
        
        Parameters
        ----------
        raw_data: np.array
            array of raw_waveforms
        sampling_interval: float
            default 4e-9
        normalize: boolean
            default True
        
        Returns
        -------
        
        '''
        self.sampling_interval = sampling_interval
        self.k = int(self.peaking_time / self.sampling_interval)
        self.l = int(self.k + (self.gap_time / self.sampling_interval))
        # self.M = int(self.decay_time / self.sampling_interval)
        self.M = 1 / (np.exp(self.sampling_interval / self.decay_time) - 1)
        d_kl = self.get_d_kl_batch(raw_data, normalize=normalize)
        d_kl[:, 0] = 0 # This makes s(0)=0
        return np.cumsum(np.cumsum(d_kl, axis=1) + d_kl * self.M, axis=1)
    
    
    def get_d_kl_batch(self, v, pretrigger_idx=200, normalize=True):
        # Batch normalization
        # TODO it would be good to experiment with different normalization scheme
        if normalize == True:
            # baseline = v[:, :pretrigger_idx].mean()
            # v = (v - baseline) / (v.max() - baseline)
            baseline = v[:, :pretrigger_idx].mean(axis=1, keepdims=True)
            v = (v - baseline) / v.max()
        # padding v by k+l with the noise mean
        noise_mean = v[:, :pretrigger_idx].mean()
        v_padded = np.pad(
            v,
            ((0, 0), (self.k + self.l, 0)),
            mode="constant",
            constant_values=(noise_mean),
            )
        # v_padded = np.pad(v, ((0, 0), (self.k + self.l, 0)), mode="symmetric")
        d_kl = (
            v_padded[:, self.k + self.l :]
            - v_padded[:, self.l : -self.k]
            - v_padded[:, self.k : -self.l]
            + v_padded[:, : -(self.k + self.l)]
            )
        assert v.shape
        return d_kl

class BogovacFilter():
    '''
    M. Bogovac, M. Jakˇsi´c, D. Wegrzynek, and A. Markowicz,
    Digital pulse processor for ion beam microprobe imaging,"
    Nuclear Instruments and Methods in Physics Research Section B:
    Beam Interactions with Materials and Atoms, vol. 267, no. 12, 
    pp. 2073{2076, Jun. 2009, doi: 10.1016/j.nimb.2009.03.033.
    
    Code by Jaewon
    '''
    def __init__(self, peaking_time, gap_time, decay_time):
        self.peaking_time = peaking_time
        self.gap_time = gap_time
        self.decay_time = decay_time
        self.k = None
        self.l = None
    
    def filter_waveform(self, raw_data, sampling_interval=4e-9, normalize=True):
        self.k = int(self.peaking_time / sampling_interval)
        self.l = int(self.k + (self.gap_time / sampling_interval))
        self.M = int(self.decay_time / sampling_interval)
        d_kl = self.get_d_kl_batch(raw_data, normalize=normalize)
        d_kl_cumsum = np.cumsum(d_kl, axis=1)
        r_n = d_kl_cumsum
        r_n[:, 1:] = r_n[:, 1:] - np.exp(-1 / self.M) * (d_kl_cumsum[:, :-1])
        return np.cumsum(r_n, axis=1)
    
    def get_d_kl_batch(self, v, pretrigger_idx=200, normalize=True):
        if normalize == True:
            baseline = v[:, :pretrigger_idx].mean(axis=1, keepdims=True)
            v = (v - baseline) / v.max()
        # padding v by k+l with the noise mean
        noise_mean = v[:, :pretrigger_idx].mean()
        v_padded = np.pad(
            v,
            ((0, 0), (self.k + self.l, 0)),
            mode="constant",
            constant_values=(noise_mean))
        d_kl = (
            v_padded[:, self.k + self.l :]
            - v_padded[:, self.l : -self.k]
            - v_padded[:, self.k : -self.l]
            + v_padded[:, : -(self.k + self.l)])
        assert v.shape == d_kl.shape
        return d_kl
class CooperFilter():
    '''
    From Ren Cooper's document and also the online blog post https://nukephysik101.wordpress.
    
    Code by Jaewon.
    '''
    def __init__(self, peaking_time, gap_time, decay_time):
        self.peaking_time = peaking_time
        self.gap_time = gap_time
        self.decay_time = decay_time
        self.k = None
        self.l = None
    
    def filter_waveform(self, raw_data, sampling_interval=4e-9, normalize=True):
        self.k = int(self.peaking_time / sampling_interval)
        self.l = int(self.k + (self.gap_time / sampling_interval))
        self.M = int(self.decay_time / sampling_interval)
        Tr_prime = self.get_Tr_prime_batch(raw_data, normalize=True)
        d_kl = self.get_d_kl_batch(Tr_prime, normalize=False)
        return np.cumsum(d_kl, axis=1)
    
    def get_Tr_prime_batch(self, v, pretrigger_idx=200, normalize=True):
        # Batch normalization
        # TODO it would be good to experiment with different normalization scheme
        if normalize == True:
            baseline = v[:, :pretrigger_idx].mean(axis=1, keepdims=True)
            v = (v - baseline) / v.max()
        # padding v by k+l with the noise mean
        noise_mean = v[:, :pretrigger_idx].mean()
        v_padded = np.pad(
            v,
            ((0, 0), (1, 0)),
            mode="constant",
            constant_values=(noise_mean))
        Tr_prime = np.cumsum(v_padded[:, 1:] - (1 - 1 / self.M) * v_padded[:, :-1], 
            axis=1)
        assert v.shape == Tr_prime.shape
        return Tr_prime
    
    def get_d_kl_batch(self, tr_prime, pretrigger_idx=200, normalize=True):
        v = tr_prime
        if normalize == True:
            baseline = v[:, :pretrigger_idx].mean()
            v = (v - baseline) / (v.max() - baseline)
        # padding v by k+l with the noise mean
        noise_mean = v[:, :pretrigger_idx].mean()
        v_padded = np.pad(
            v,
            ((0, 0), (self.k + self.l, 0)),
            mode="constant",
            constant_values=(noise_mean))
        d_kl = (
            v_padded[:, self.k + self.l :]
            - v_padded[:, self.l : -self.k]
            - v_padded[:, self.k : -self.l]
            + v_padded[:, : -(self.k + self.l)])
        assert v.shape == d_kl.shape
        return d_kl

def exponential(t, a, tau):
    return a * np.exp(-t / tau)

def fit_tau(waveform, 
        pre_sample_length=1100, 
        fit_length = 48000,
        show_plot=False,
        plot_save_name=None):
    '''
    Find time constant (tau) of single waveform
    
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
        plt.ylabel('Magnitude (ADC Units)')
        plt.text(3800,25,r'$\tau$='+str(round(tau,2)))
        plt.legend()
        plt.show()
        if plot_save_name is not None:
            plt.savefig(plot_save_name)
    return tau

def fit_taus(waveforms, 
        pre_sample_length=1100, 
        fit_length = 3800):
    '''
    Find time constant (tau) of multiple waveforms
    
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
    taus: np.array
        array of fitted tau values for waveforms
    '''
    # initialize tau array
    taus = np.zeros(len(waveforms))
    
    # loop through waveforms
    for i in range(len(waveforms)):
        taus[i] = fit_tau(waveforms[i],pre_sample_length=pre_sample_length,fit_length=fit_length)
    print('mean and variance:',taus.mean(),taus.var())
        
    return taus
    