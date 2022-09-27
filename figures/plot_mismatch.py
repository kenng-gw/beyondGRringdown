from matplotlib import pyplot as plt
import h5py as h5
from matplotlib.colors import LogNorm
import numpy as np
import qnm
from latex import *
import argparse

p = argparse.ArgumentParser(description="Generate dCS waveform for a given coupling parameter value")
p.add_argument("--inputdir", default="../dCS_mms", type=str,help="input path")
p.add_argument("--outdir", default="./", type=str,help="output path")
p.add_argument("--Ntones", default=8, type=int,help="no of tones (including fundamental)")
args = p.parse_args()

Ntones = args.Ntones
outdir = args.outdir
def plot_mm(mms, chi_fs, M_fs, chi_f_true, M_f_true, outname):
	plt.figure()
	plt.ylabel(r"$\chi_f$")
	plt.xlabel(r"$M_f$")
	plt.scatter(M_fs, chi_fs, c=mms, norm=LogNorm(vmin=1e-7, vmax=1e-3))
	plt.axvline(x=M_f_true, c='white', ls='dotted')
	plt.axhline(y=chi_f_true, c='white', ls='dotted')
	plt.colorbar()
	#plt.title(r"(M_f = %.4f, chi_f = %.4f)" %(M_fs[mms==mms.min()][0], chi_fs[mms==mms.min()][0]))
	plt.savefig(outname)
	plt.close()

#chi_f_true_dCS, M_f_true_dCS = 0.692085186818, 0.952032939704
#def get_GR_qnms(chi_f, M_f, l=2, m=2, s=-2):
#	omegas_GR, taus_GR=[],[]
#	for n in range(Ntones):
#		qnm_mode = qnm.modes_cache(s=-2,l=2,m=2,n=n)
#		omega_complex, _, _ = qnm_mode(a=chi_f)
#		omega_complex /= M_f
#		omega = np.real(omega_complex)
#		tau=np.abs(1./np.imag(omega_complex))
#		omegas_GR.append(omega)
#		taus_GR.append(tau)
#	return omegas_GR, taus_GR
#
#omegas_GR_dCS, taus_GR_dCS = get_GR_qnms(chi_f_true_dCS, M_f_true_dCS, l=2, m=2, s=-2)
def plot_amp_t(ts, As, outname):
	plt.figure()
	plt.xlabel(r"$t$")
	plt.ylabel(r"$A_{22n}$")
	for (i,A) in enumerate(As):
		plt.plot(ts, A, label=r"$%d$" %i)#*np.exp(-ts/taus_GR_dCS[i])
	plt.legend()
	plt.ylim(0.01, 100)
	plt.yscale('log')
	plt.savefig(outname)
	plt.close()

def plot_mm_t(mms_list, ts_list, label_list, outname):
	plt.figure()
	plt.xlabel(r"$t$")
	plt.ylabel(r"$\mathcal{M}$")
	for (mms, ts, label) in zip(mms_list, ts_list, label_list):
		plt.semilogy(ts, mms, label=label)
	#plt.axvline(x=M_f_true, c='white')
	#plt.axhline(y=chi_f_true, c='white')
	#plt.colorbar()
	#plt.title("(M_f = %.4f, chi_f = %.4f)" %(M_fs[mms==mms.min()][0], chi_fs[mms==mms.min()][0]))
	plt.legend()
	plt.ylim(1e-7, 1)
	plt.savefig(outname)
	plt.close()

mms_list, ts_list, label_list = [], [], []
for ell in [0.3,0.226,0.2,0.1,0.0]:
	ell_string=str(ell).replace(".","p")
	cat = h5.File(args.inputdir+"/mismatch_t_n%d_ell%s_real.hdf5" %(Ntones,ell_string), 'r')
	ts = np.copy(cat["t"][()])
	mms = np.copy(cat["mismatch"][()])
	mms_list.append(mms)
	ts_list.append(ts)
	if ell!=0.226 and ell!=0.0:
		label_list.append(r"$\ell=%.1f$" %ell)
	elif ell==0.226:
		label_list.append(r"$\ell=%.3f$" %ell)
	elif ell==0.0:
		label_list.append(r"GR")
	cat.close()

plot_mm_t(mms_list, ts_list, label_list, outdir+"/mismatch_t_n%d_real.pdf" %Ntones)

#chi_f_true, M_f_true = 0.692, 0.9525
chi_f_true, M_f_true = 0.692085186818, 0.952032939704

for ell in [0.3,0.226,0.2,0.1,0.0]:
	ell_string=str(ell).replace(".","p")
	cat = h5.File(args.inputdir+"/mismatch_M_chi_n%d_ell%s_real.hdf5" %(Ntones,ell_string), 'r')
	mms, chi_fs, M_fs = cat["mismatch"][()], cat["chi_f"][()], cat["M_f"][()]
	plot_mm(mms, chi_fs, M_fs,chi_f_true, M_f_true, outdir+"/ell%s_mismatch_M_chi_n%d_real.pdf" %(ell_string,Ntones))

	cat.close()
