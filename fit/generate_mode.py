# imports
from Generate_dCS_Strain import *
import numpy as np
from lal import MTSUN_SI, PC_SI, C_SI, SpinWeightedSphericalHarmonic
from gwpy.timeseries import TimeSeries

# ## Helper functions
def CutTimes(time, data, TLow, TUp): 
    """ Cut time and data to be between 
        TLow and TUp  """
    TLowIndex = np.where(time >= TLow)[0][0]
    TUpIndex = np.where(time <= TUp)[0][-1]
    time_out = np.copy(time[TLowIndex:TUpIndex])
    data_out = np.copy(data[TLowIndex:TUpIndex])
    return time_out, data_out

def GetPeakTimeMode(time, data): 
    """ Grab the peak time of some data """
    t_peak = time[np.argmax(np.absolute(data))]
    return t_peak

def SubtractPeakTimeMode(time, data): 
    """ Subtract the peak time of some data """
    t_peak = GetPeakTimeMode(time, data)
    return time - t_peak

# ## Plot SXS waveforms 
all_modes = []
for l in range(2, 9, 1):
	for m in range(-l, l+1, 1):
		all_modes.append((l,m))

def dCS_hlm(ell, mode, start=1000, stop=10000, **kwargs):
	p = "Waveforms/"
	time, strain = ReadExtrapolatedMode(p, "dCSModified", mode, ell=ell)
	time_out, strain_out = CutTimes(time, strain, start, stop)
	return time_out, strain_out

def SXS_hlm(ell, mode, start=1000, stop=10000, **kwargs):
	p = "Waveforms/"
	time, strain = ReadExtrapolatedMode(p, "SXSGR", mode, ell=ell)
	time_out, strain_out = CutTimes(time, strain, start, stop)
	return time_out, strain_out

def ascii_hlm(name):
	data = np.genfromtxt(name)
	time, hp, hc = data[0], data[1], data[2]
	h = hp-1j*hc
	return time, h

def get_ringdown(time, strain):
	t_peak = GetPeakTimeMode(time, strain)
	return time[time-t_peak>0]-t_peak, strain[time-t_peak>0]

