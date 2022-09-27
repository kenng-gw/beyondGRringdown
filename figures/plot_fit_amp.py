from matplotlib import pyplot as plt
import h5py as h5
from matplotlib.colors import LogNorm
import numpy as np
import qnm
from latex import *
import argparse

outdir = "/home/kwan-yeung.ng/public_html/dCSringdown/figures/inference/"
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

ts_list = []
for ell in [0.0]:#[0.3,0.226,0.2,0.1,0.0]:
	ell_string=str(ell).replace(".","p")
	for N in range(8):
		cat_dCS = h5.File("mismatch_t_dCSn%d_ell%s_real.hdf5" %((N+1),ell_string), 'r')
		ts = np.copy(cat_dCS["t"][()])
		As = np.abs(np.copy(cat_dCS["cs"][()])).T
		plot_amp_t(ts, As, outdir+"/tones_amp/ell%s/tone_amplitudes_t_dCSn%d.pdf" %(ell_string,N+1))
		cat_dCS.close()
	ts_list.append(ts)
