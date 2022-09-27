import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import lstsq, pinv, pinv2
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from generate_modes import *
import qnm
import h5py as h5
from scipy.sparse import linalg as splinalg
import argparse
import multiprocessing

p = argparse.ArgumentParser(description="Generate dCS waveform for a given coupling parameter value")
p.add_argument("--ell", required=True, type=float,help="Value of dCS coupling constant")
p.add_argument("--Ntones", default=8, type=int,help="no of tones (including fundamental)")
p.add_argument("--fit_tpeak", default=False, action="store_true")
p.add_argument("--fit_chiM", default=False, action="store_true")
args = p.parse_args()

Ntones = args.Ntones
def get_GR_qnms(chi_f, M_f, l=2, m=2, s=-2):
	omegas_GR, taus_GR=[],[]
	for n in range(Ntones):
		qnm_mode = qnm.modes_cache(s=-2,l=2,m=2,n=n)
		omega_complex, _, _ = qnm_mode(a=chi_f)
		omega_complex /= M_f
		omega = np.real(omega_complex)
		tau=np.abs(1./np.imag(omega_complex))
		omegas_GR.append(omega)
		taus_GR.append(tau)
	return omegas_GR, taus_GR

def inner_product_trapz(a, b, t):
	integrand = a*np.conj(b)
	real_integral = np.trapz(np.real(integrand), x=t)
	imag_integral = np.trapz(np.imag(integrand), x=t)
	return real_integral + 1j*imag_integral

def inner_product_dot(a, b, t):
	dt = t[1]-t[0]
	return np.dot(a,np.conj(b))*dt

def mismatch(htrue,hfit,t,inner_method="dot",mm_method="real"):
	if inner_method=="trapz":
		inner_product=inner_product_trapz
	elif inner_method=="dot":
		inner_product=inner_product_dot
	htrue_hfit = inner_product(htrue, hfit,t)
	htrue_htrue = inner_product(htrue, htrue,t)
	hfit_hfit = inner_product(hfit, hfit,t)
	if mm_method=="absolute":
		return 1.-np.abs(htrue_hfit)/np.sqrt(np.abs(htrue_htrue)*np.abs(hfit_hfit))
	elif mm_method=="real":
		return 1.-np.real(htrue_hfit)/np.sqrt(np.real(htrue_htrue)*np.real(hfit_hfit))

# Prepare waveform
def get_waveform(ell, t_start=-50.0, mode=(2,2), generator=SXS_hlm):
	time, strain = generator(ell,mode,start=t_start,end=90.0)
	t_vec = np.linspace(time[0], time[-1], int((time[-1]-time[0])/0.01)+1)
	spline_re = splrep(time, np.real(strain))
	spline_im = splrep(time, np.imag(strain))
	h_RD_real = splev(t_vec, spline_re)
	h_RD_imag = splev(t_vec, spline_im)
	h_RD = h_RD_real+1.j*h_RD_imag
	h_RD -= h_RD[-1]
	t_peak = GetPeakTimeMode(t_vec, h_RD)
	t_vec = t_vec-t_peak
	return t_vec[t_vec>=t_start], h_RD[t_vec>=t_start]

def get_waveform_luis(name):
	data = np.genfromtxt(name, unpack=True)
	time, hp, hc = data[0], data[1], data[2]
	strain = hp-1j*hc
	t_vec = np.linspace(time[0], time[-1], int((time[-1]-time[0])/0.01)+1)
	spline_re = splrep(time, np.real(strain))
	spline_im = splrep(time, np.imag(strain))
	h_RD_real = splev(t_vec, spline_re)
	h_RD_imag = splev(t_vec, spline_im)
	h_RD = h_RD_real+1.j*h_RD_imag
	h_RD -= h_RD[-1]
	t_peak = GetPeakTimeMode(t_vec, h_RD)
	t_vec = t_vec-t_peak
	return t_vec[t_vec>=t_start], h_RD[t_vec>=t_start]


t_start = -50.0

# Prepare dCS waveform
ell=args.ell
if ell==0.0:
	t_, h_RD_22_ = get_waveform(ell, generator=dCS_hlm)
	t_, h_RD_2m2_ = get_waveform(ell, mode=(2,-2), generator=dCS_hlm)
elif ell>0:
	t_, h_RD_22_ = get_waveform(ell, generator=dCS_hlm)
	t_, h_RD_2m2_ = get_waveform(ell, mode=(2,-2), generator=dCS_hlm)
else:
	t_, h_RD_22_ = get_waveform_luis(name="luis_wfs/wavesout_uni_m10_e1_p4.dat")	
	h_RD_2m2_ = np.conj(h_RD_22_)

t_end = 90.0
h_RD_22 = h_RD_22_[t_<=t_end]
h_RD_2m2 = h_RD_2m2_[t_<=t_end]
t = t_[t_<=t_end]
"""
mm_chiral=mismatch(h_RD_22, np.conj(h_RD_2m2), t)
print("mismatch between 22 and 2-2 is %g" %mm_chiral)
"""

def fit_ntones(n, omegas, taus, t_start, t_RD_raw, h_RD_raw):
	t_RD = t_RD_raw[t_RD_raw>=t_start]
	h_RD = h_RD_raw[t_RD_raw>=t_start]
	out=[]
	for (tau, omega) in zip(taus, omegas):
		tone_re = np.exp(-(t_RD-t_RD[0])/tau)*np.cos(omega*(t_RD-t_RD[0]))
		tone_im = -1.j*np.exp(-(t_RD-t_RD[0])/tau)*np.sin(omega*(t_RD-t_RD[0]))
		tone = tone_re + tone_im
		out.append(tone)
	Amatrix = np.transpose(out)
	cs, ress, rank, sing = np.linalg.lstsq(Amatrix,h_RD,rcond=1e-50) #lstsq(Amatrix,h_RD)#
	#cs, _, rank, _ = lstsq(Amatrix, h_RD, cond=1e-100, lapack_driver = "gelss")
	#cs, _, _, _, _, _, _, _, _, _ = splinalg.lsqr(Amatrix,h_RD,atol=0,btol=0,conlim=0, x0=[0.97, 4.22, 11.3, 23.0, 33., 29., 14., 2.9]) #lstsq(Amatrix,h_RD)#
	#cs = splinalg.spsolve(Amatrix, h_RD)
	h_RD_fit = np.dot(Amatrix, cs)
	mm_real = mismatch(h_RD, h_RD_fit, t_RD, mm_method="real")
	mm_abs = mismatch(h_RD, h_RD_fit, t_RD, mm_method="absolute")
	return mm_real, mm_abs, t_RD, h_RD, h_RD_fit, cs

string = 'ell' + str(ell).replace('.', 'p')
if args.fit_tpeak is True:
	# get GR tones
	#SXS: 0.692085186818, 0.952032939704
	#dCS: 0.692, 0.9525
	chi_f_true, M_f_true = 0.692085186818, 0.952032939704
	#step=0.001
	#bound=step*50
	#chi_f_vec, M_f_vec = np.arange(chi_f_true-bound, chi_f_true+bound, step), np.arange(M_f_true-bound, M_f_true+bound, step)#[chi_f_true], [M_f_true]#
	ts = np.arange(-20.0, 50.1, 0.1)
	mms_t_real, mms_t_abs = np.zeros(ts.size), np.zeros(ts.size)
	cs_t = np.zeros((ts.size, Ntones))+1.j
	
	omegas_GR, taus_GR = get_GR_qnms(chi_f_true, M_f_true, l=2, m=2, s=-2)
	def get_mms_tpeak(index):
		t0 = ts[index]
		mm_real, mm_abs, t_RD, h_RD, h_RD_fit, cs = fit_ntones(Ntones, omegas_GR, taus_GR, t0, t, h_RD_22)
		return [mm_real, mm_abs, cs]
	
	pool = multiprocessing.Pool(16)
	for (i,p) in enumerate(pool.imap(get_mms_tpeak, range(ts.size))):
		mms_t_real[i]=p[0]
		mms_t_abs[i]=p[1]
		cs_t[i] = p[2]
	
	pool.close()
	
	def save_mms(outname, mms, ts, cs):
		cat=h5.File(outname,'w')
		cat.create_dataset(name="mismatch", data=mms)
		cat.create_dataset(name="t", data=ts)
		cat.create_dataset(name="cs", data=cs)
		cat.close()
	
	save_mms("mismatch_t_n%d_%s_real.hdf5" %(Ntones, string), mms_t_real, ts, cs_t)
	#save_mms("mismatch_tn%d_ell%s_abs.hdf5" %(Ntones, ell_string), mms_t_abs, ts, cs_t)

if args.fit_chiM is True:
	chi_f_vec, M_f_vec = np.linspace(0.0, 0.9, 101), np.linspace(0.85, 1.0, 101)
	chi_f_grid, M_f_grid = np.meshgrid(chi_f_vec, M_f_vec)
	chi_fs, M_fs = chi_f_grid.flatten(), M_f_grid.flatten()
	mms_Mchi_real, mms_Mchi_abs, cs_Mchi = np.zeros(chi_fs.size), np.zeros(chi_fs.size), np.zeros((chi_fs.size, Ntones))+1.j
	
	def get_mms(index):
		chi_f, M_f = chi_fs[index], M_fs[index]
		omegas_GR, taus_GR = get_GR_qnms(chi_f, M_f, l=2, m=2, s=-2)
		mm_real, mm_abs, t_RD, h_RD, h_RD_fit, cs = fit_ntones(Ntones, omegas_GR, taus_GR, 0, t, h_RD_22)
		return [mm_real, mm_abs, cs]
	
	pool = multiprocessing.Pool(16)
	for (i,p) in enumerate(pool.imap(get_mms, range(chi_fs.size))):
		mms_Mchi_real[i]=p[0]
		mms_Mchi_abs[i]=p[1]
		cs_Mchi[i]=p[2]
	
	pool.close()
	
	def save_mms(outname, mms, chi_fs, M_fs, cs):
		cat=h5.File(outname,'w')
		cat.create_dataset(name="mismatch", data=mms)
		cat.create_dataset(name="chi_f", data=chi_fs)
		cat.create_dataset(name="M_f", data=M_fs)
		cat.create_dataset(name="cs", data=cs)
		cat.close()
	
	save_mms("mismatch_M_chi_n%d_%s_real.hdf5" %(Ntones, string), mms_Mchi_real, chi_fs, M_fs, cs_Mchi)
