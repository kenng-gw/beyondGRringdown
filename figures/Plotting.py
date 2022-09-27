# imports
from matplotlib import pyplot as plt
import matplotlib as mpl
from Generate_dCS_Strain import *
import numpy as np
from latex import *
from scipy.interpolate import interp1d
from lal import MTSUN_SI, PC_SI, C_SI, SpinWeightedSphericalHarmonic
from gwpy.timeseries import TimeSeries
from astropy.cosmology import Planck15
from scipy.integrate import cumtrapz

Mpc = 1.e6*PC_SI
t_solar = MTSUN_SI
c_speed = C_SI

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
#all_modes=[(2,2), (2,-2)]

def GetPlusCrossStrainElls(ell, Mtot, dL, theta_jn, phase, t_peak = None, modes=[(2,2), (2,-2)], **kwargs):
	""" Plot the strains generated with Generate_dCS_Strain.py
	    ells is an array of values of the coupling constant 
	    (note that these waveforms need to already exist )"""
	
	p = "Waveforms/"
	start = -2000
	stop = 150
	
	strain = 0.0+0.0j
	for mode in modes:
		time, strain_out = ReadExtrapolatedMode(p, "dCSModified", mode, ell=ell)
		time_tmp = SubtractPeakTimeMode(time, strain_out) if t_peak is None else time-t_peak
		time_tmp, strain_tmp = CutTimes(time_tmp, strain_out, start, stop)
		time_tmp *= Mtot*t_solar
		strain_tmp *= Mtot*t_solar/(dL*Mpc/c_speed)
		"""
		f_tmp, ax_tmp = plt.subplots(1, figsize=(14,8))
		ax_tmp.plot(time_tmp, np.real(strain_tmp))
		#legend = ax_tmp.legend(fontsize=24, loc='upper left', frameon=False, ncol=1, title='$\ell/GM$')
		ax_tmp.set_ylabel('$h(t)\,(%d,%d)$' %(mode[0],mode[1]), fontsize=30)
		ax_tmp.set_xlabel('$(t - t_\mathrm{peak}) (s)$', fontsize=30) 
		ax_tmp.set_xlim(-50*Mtot*t_solar, 40*Mtot*t_solar)
		f_tmp.tight_layout()
		f_tmp.savefig('./figures/%d_%d_ell_%s.pdf' %(mode[0],mode[1],str(ell).replace('.', 'p')))
		plt.close(f_tmp)
		"""
		strain += strain_out*SpinWeightedSphericalHarmonic(theta_jn, phase, -2, mode[0], mode[1])
	if t_peak is None:
		t_peak = GetPeakTimeMode(time, strain)
	time = time - t_peak
	time, strain = CutTimes(time, strain, start, stop)
	time *= Mtot*t_solar
	strain *= Mtot*t_solar/(dL*Mpc/c_speed)
	#print(np.max(np.abs(np.real(strain))))
	#print(np.max(np.abs(np.imag(strain))))
	return strain, time, t_peak

z=0.5
dL = 400#Planck15.luminosity_distance(z).value
Mtot = 60.#*(1+z)
start = -2000
stop = 200
#outname_plus="hplus_all_modes_Mtot_60Msun_dL_400Mpc_ell_%s" %(str(ell).replace('.', 'p'))
#outname_cross="hcross_all_modes_Mtot_60Msun_dL_400Mpc_ell_%s" %(str(ell).replace('.', 'p'))

outname_plus="hplus_all_modes_Mtot_60Msun_dL_400Mpc"
outname_cross="hcross_all_modes_Mtot_60Msun_dL_400Mpc"
f_plus, ax_plus = plt.subplots(1, figsize=(14,8))
f_cross, ax_cross = plt.subplots(1, figsize=(14,8))
t_peak = None#2553.6978
strain_GR, time_GR = GetComplexStrainElls(0.0, Mtot, dL, 0, 0, t_peak = t_peak)
#strain, time = GetComplexStrainElls(ax_plus, ax_cross, 0.2, Mtot, dL, 0, 0, t_peak = t_peak)
strain_dCS, time_dCS = GetComplexStrainElls(0.3, Mtot, dL, 0, 0, t_peak = t_peak)
#PlotStrainElls(ax_plus, ax_cross, ell, Mtot, dL, np.pi/2.0, t_peak = t_peak, c='C1', label=r'$\theta_{JN}=\pi/2$')

## Color, linewidth, and linestyle values -- please modify as needed 
colors = ['blue', 'lightblue', 'red', 'black']
lss = ['-','--','-', '--']
lws = [4.0, 4.0, 2.0, 2.0]
ax_plus.plot(time_dCS, np.real(strain_dCS))
ax_cross.plot(time_dCS, -np.imag(strain_dCS))

ax_plus.legend()
ax_plus.set_ylabel(r'$h_{+}(t)$', fontsize=30)
ax_plus.set_xlabel(r'$(t - t_\mathrm{peak})$ (s)', fontsize=30) 
ax_plus.set_xlim(-50*Mtot*t_solar, 40*Mtot*t_solar)
ax_plus.set_ylim(-2.1e-21, 2.1e-21)
f_plus.savefig('./figures/%s.pdf' %outname_plus)
ax_cross.legend()
ax_cross.set_ylabel(r'$h_{\times}(t)$', fontsize=30)
ax_cross.set_xlabel(r'$(t - t_\mathrm{peak})$ (s)', fontsize=30) 
ax_cross.set_xlim(-50*Mtot*t_solar, 40*Mtot*t_solar)
ax_cross.set_ylim(-2.1e-21, 2.1e-21)
f_cross.savefig('./figures/%s.pdf' %outname_cross)
ax_plus.set_xlim(-2000*Mtot*t_solar, 150*Mtot*t_solar)
f_plus.savefig('./figures/%s_full.pdf' %outname_plus)
ax_cross.set_xlim(-2000*Mtot*t_solar, 150*Mtot*t_solar)
f_cross.savefig('./figures/%s_full.pdf' %outname_cross)

plt.close()

t_vec_GR = np.linspace(time_GR[0], time_GR[-1], 8192)
t_vec_dCS = np.linspace(time_dCS[0], time_dCS[-1], 8192)
hp_GR = interp1d(time_GR, np.real(strain_GR))(t_vec_GR)
hp_dCS = interp1d(time_dCS, np.real(strain_dCS))(t_vec_dCS)
hplus_GR = TimeSeries(hp_GR, times=t_vec_GR).asd()
hplus_dCS = TimeSeries(hp_dCS, times=t_vec_dCS).asd()
CE_ASD_DATA = np.genfromtxt("ce_asd.txt",unpack=True)
CE_asd = interp1d(CE_ASD_DATA[0], CE_ASD_DATA[1], bounds_error=False, fill_value=1.0)
def SNR(f, hf1, hf2, Sn_f):
	integrand=hf1*np.conj(hf2)/Sn_f
	return np.sqrt(np.real(np.trapz(integrand, x=f)*4*2))

def SNR_cum(f, hf1, hf2, Sn_f):
	integrand=hf1*np.conj(hf2)/Sn_f
	return np.sqrt(np.real(cumtrapz(integrand, x=f, initial=0.0)*4*2))

hp_GR = interp1d(hplus_GR.frequencies, hplus_GR.value, bounds_error=False, fill_value=0.0)
hp_dCS = interp1d(hplus_dCS.frequencies, hplus_dCS.value, bounds_error=False, fill_value=0.0)
freq_vec = np.arange(125.,1000.,1./8.)
hdCS_hdCS = SNR(freq_vec, hp_dCS(freq_vec), hp_dCS(freq_vec), CE_asd(freq_vec)**2)
hGR_hGR = SNR(freq_vec, hp_GR(freq_vec), hp_GR(freq_vec), CE_asd(freq_vec)**2)
hdCS_hGR = SNR(freq_vec, hp_dCS(freq_vec), hp_GR(freq_vec), CE_asd(freq_vec)**2)
mismatch = 1.-hdCS_hGR/np.sqrt(hGR_hGR*hdCS_hdCS)
print(mismatch)
print(hGR_hGR)
print(hdCS_hdCS)
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$\tilde{h}_{+}$")
plt.xlabel(r"$f \mathrm{(Hz)}$")
plt.xlim(40, 1024)
plt.plot(hplus_GR.frequencies, hplus_GR.value, label=r'GR')
plt.plot(hplus_dCS.frequencies, hplus_dCS.value, label=r'dCS $\ell=0.3$')
freq_vec = np.arange(5.,1000.,1./8.)
plt.plot(freq_vec, CE_asd(freq_vec))
plt.savefig("./figures/fft_check.pdf")
plt.close()

freq_vec = np.arange(40.,1000., 1./8.)
hdCS_hdCS_cum = SNR_cum(freq_vec, hp_dCS(freq_vec), hp_dCS(freq_vec), CE_asd(freq_vec)**2)
hGR_hGR_cum = SNR_cum(freq_vec, hp_GR(freq_vec), hp_GR(freq_vec), CE_asd(freq_vec)**2)
hGR_hGR_cum_inv = np.sqrt(hGR_hGR_cum[-1]**2-hGR_hGR_cum**2)
hdCS_hdCS_cum_inv = np.sqrt(hdCS_hdCS_cum[-1]**2-hdCS_hdCS_cum**2)
plt.figure()
plt.xscale('log')
plt.ylabel(r'$\rho_{\mathrm{dCS}}(f_{\mathrm{min}}):\rho_{\mathrm{GR}}(f_{\mathrm{min}})$')
plt.xlabel(r"$f_{\mathrm{min}} \mathrm{(Hz)}$")
#plt.plot(freq_vec, hGR_hGR_cum_inv, label=r'GR')
freq = np.geomspace(40., 400., 101)
hGR_hGR_cum_inv_sc = interp1d(freq_vec,hGR_hGR_cum_inv)(freq)
plt.scatter(freq, interp1d(freq_vec,hdCS_hdCS_cum_inv/hGR_hGR_cum_inv)(freq), c=hGR_hGR_cum_inv_sc, norm=mpl.colors.LogNorm(vmax=hGR_hGR_cum_inv_sc.max(), vmin=hGR_hGR_cum_inv_sc.min()))
cb=plt.colorbar()
cb.ax.set_xlabel(r'$\rho_{\mathrm{GR}}(f_{\mathrm{min}})$')
plt.xlim(freq[0], freq[-1])
plt.savefig("./figures/snr_evol.pdf")
plt.close()

