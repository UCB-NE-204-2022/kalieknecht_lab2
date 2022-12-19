import numpy as np
import matplotlib.pyplot as plt

def find_rise_time(waveform,
    pretrigger=[0,500],
    signal_end = 1000,
    max_signal_window=10,
    plotting=False):
    '''
    Parameters
    ----------
    waveform: ndarray
        raw waveform
    pretrigger: list
        start, end index of pretrigger background
    signal_end: int
        index of signal end
    max_signal_window: int
        window length to find mean of for max signal
        
    Returns
    -------
    rise_time: number
        rise time in s
    
    '''
    # separate out background and signal based on pretrigger
    background = waveform[pretrigger[0]:pretrigger[1]]
    
    # only want rise of first pulse
    # cut at signal_end to avoid finding max on piled up pulse
    signal = waveform[pretrigger[1]:pretrigger[1]+signal_end]
    
    # find mean average background    
    ave_bkg = background.mean()
    
    # signal - background
    bkg_corrected_signal = signal - ave_bkg
    
    # find max signal
    max_signal = bkg_corrected_signal.max()
    
    # find mean max signal within some window to account for fluctuations
    max_signal_index = np.argwhere(bkg_corrected_signal == max_signal).flatten()[0]
    max_signal = bkg_corrected_signal[max_signal_index:max_signal_index+max_signal_window].mean()

    # first value to reach 10% of max
    t10 = np.argwhere(bkg_corrected_signal >= max_signal * 0.1)[0]
    
    # first value at 90% of max
    t90 = np.argwhere(bkg_corrected_signal >= max_signal * 0.9)[0]
    
    # find rise time and convert to s
    # leave in nanoseconds?
    rise_time = (t90 - t10) * 4 * 10 ** -9
    
    if plotting:
        return (rise_time, max_signal, ave_bkg, t10, t90)
    else:
        return rise_time
    
def plot_rise_time(waveform,
    plot_window=100,
    pretrigger=[0,500],
    signal_end = 1000,
    max_signal_window=10):
    '''
    Plot rise time calculation on top of raw waveform
    
    Parameters
    ----------
    waveform: ndarray
        raw waveform
    pretrigger: list
        start, end index of pretrigger background
    signal_end: int
        index of signal end
    max_signal_window: int
        window length to find mean of for max signal
        
    Returns
    -------
    plot: matplotlib.pyplot.figure
        plot of spectrum with energy in keV on x-axis
    '''
    # calc rise time
    rise_time, max_signal, ave_bkg, t10, t90 = find_rise_time(waveform,
        pretrigger=pretrigger,
        signal_end=signal_end,
        max_signal_window=max_signal_window,
        plotting=True)
    
    # plot
    plt.figure()
    
    plt.plot(waveform)

    plt.axhline(max_signal + ave_bkg,label='Max Signal',c='tab:orange')
    plt.axvline(t10+pretrigger[1],label='10% of max',c='tab:green')
    plt.axvline(t90+pretrigger[1],label='90% of max',c='tab:red')
    plt.text(t10+pretrigger[1]-plot_window+5,ave_bkg+20,r'rise time = {:.1f}'.format(rise_time[0]*10**9)+' ns')
    # formatting
    plt.ylabel('Magnitude (ADC Units)')
    plt.xlabel('Measurement Time (Clock Cycles)')
    plt.legend(fontsize=7)
    plt.xlim(t10+pretrigger[1]-plot_window,t90+pretrigger[1]+plot_window)
    plt.show()
    
    
    
    