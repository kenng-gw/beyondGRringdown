from pylab import *
import h5py
import lal
import lalsimulation as lalsim
import scipy.signal
from scipy.interpolate import splrep, splev
import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], "lscsoft/src/lalsuite-extra/data/lalsimulation/")
import ringdown as rd
from Generate_dCS_Strain import *

def get_tgps_aps(tgps_geocent, ra, dec, psi=0, ifos=('H1', 'L1')):
    """ Returns times of arrival and antenna patterns at different detectors 
    for a given geocenter triggertime and source location/orientation.
    
    Arguments
    ---------
    tgps_geocent: float
        GPS time of arrival at geocenter
    ra: float
        source right ascension
    dec: float
        source declination
    psi: float
        polarization angle (def: 0)
    ifos: list of strs
        detector names (def: ['H1', 'L1'])
    
    Returns
    -------
    tgps_dict: dict
        dictionary of arrival GPS times at each detector
    ap_dict: dict
        dictionary of antenna patterns at each detector
    """
    lal_t_geo = lal.LIGOTimeGPS(tgps_geocent)
    gmst = lal.GreenwichMeanSiderealTime(lal_t_geo)
    tgps_dict = {}
    ap_dict = {}
    for ifo in ifos:
        d = lal.cached_detector_by_prefix[ifo]
        dt_ifo = lal.TimeDelayFromEarthCenter(d.location, ra, dec, lal_t_geo)
        tgps_dict[ifo] = tgps_geocent + dt_ifo
        ap_dict[ifo] = lal.ComputeDetAMResponse(d.response, ra, dec, psi, gmst)
    return tgps_dict, ap_dict

def generate_lal_hphc(approximant_key, m1_msun=None, m2_msun=None, chi1=None,
                      chi2=None, dist_mpc=None, dt=None, f_low=20, f_ref=None,
                      inclination=None, phi_ref=None, ell_max=None,
                      single_mode=None, epoch=None, mtot_msun=None, 
                      nr_path=None):

    approximant = lalsim.SimInspiralGetApproximantFromString(approximant_key)

    param_dict = lal.CreateDict()
    
    if f_ref is None:
        f_ref = f_low

    # NR handling based on https://arxiv.org/abs/1703.01076
    if approximant_key == 'NR_hdf5':
        # get masses
        mtot_msun = mtot_msun or m1_msun + m2_msun
        with h5py.File(nr_path, 'r') as f:
            m1 = f.attrs['mass1']
            m2 = f.attrs['mass2']
            m1_msun = m1 * mtot_msun/(m1 + m2)
            m2_msun = m2 * mtot_msun/(m1 + m2)
        # Compute spins in the LAL frame
        s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(f_ref, mtot_msun, nr_path)
        chi1 = [s1x, s1y, s1z]
        chi2 = [s2x, s2y, s2z]
        # Create a dictionary and pass /PATH/TO/H5File
        lalsim.SimInspiralWaveformParamsInsertNumRelData(param_dict, nr_path)
        longAscNodes = np.pi / 2
    else:
        longAscNodes = 0.

    m1_kg = m1_msun*lal.MSUN_SI
    m2_kg = m2_msun*lal.MSUN_SI
    
    distance = dist_mpc*1e6*lal.PC_SI

    if single_mode is not None and ell_max is not None:
        raise Exception("Specify only one of single_mode or ell_max")

    if ell_max is not None:
        # If ell_max, load all modes with ell <= ell_max
        ma = lalsim.SimInspiralCreateModeArray()
        for ell in range(2, ell_max+1):
            lalsim.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
        lalsim.SimInspiralWaveformParamsInsertModeArray(param_dict, ma)
    elif single_mode is not None:
        # If a single_mode is given, load only that mode (l,m) and (l,-m)
        param_dict = set_single_mode(param_dict, single_mode[0], single_mode[1])

    hp, hc = lalsim.SimInspiralChooseTDWaveform(m1_kg, m2_kg,
                                                chi1[0], chi1[1], chi1[2],
                                                chi2[0], chi2[1], chi2[2],
                                                distance, inclination,
                                                phi_ref, longAscNodes,
                                                0., 0., dt, f_low, f_ref,
                                                param_dict, approximant)
    return hp, hc

def generate_lal_waveform(*args, **kwargs):
    times = kwargs.pop('times')
    triggertime = kwargs.pop('triggertime')
    manual_epoch = kwargs.pop('manual_epoch', False)
    
    bufLength = len(times)
    delta_t = times[1] - times[0]
    tStart = times[0]
    tEnd = tStart + delta_t * bufLength

    kwargs['dt'] = delta_t

    hplus = kwargs.pop('hplus', None)
    hcross = kwargs.pop('hcross', None)
    if (hplus is None) or (hcross is None):
        hplus, hcross = generate_lal_hphc(*args, **kwargs)
    
    # align waveform, based on LALInferenceTemplate
    # https://git.ligo.org/lscsoft/lalsuite/blob/master/lalinference/lib/LALInferenceTemplate.c#L1124

    # /* The nearest sample in model buffer to the desired tc. */
    tcSample = round((triggertime - tStart)/delta_t)

    # /* The actual coalescence time that corresponds to the buffer
    #    sample on which the waveform's tC lands. */
    # i.e. the nearest time in the buffer
    injTc = tStart + tcSample*delta_t

    # /* The sample at which the waveform reaches tc. */
    if manual_epoch:
        # manually find peak of the waveform envelope
        habs = np.sqrt(hplus.data.data**2 + hcross.data.data**2)
        waveTcSample = np.argmax(habs)
    else:
        hplus_epoch = hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds*1E-9
        waveTcSample = round(-hplus_epoch/delta_t)

    # /* 1 + (number of samples post-tc in waveform) */
    wavePostTc = hplus.data.length - waveTcSample

    # bufStartIndex = (tcSample >= waveTcSample ? tcSample - waveTcSample : 0);
    bufStartIndex = int(tcSample - waveTcSample if tcSample >= waveTcSample else 0)
    # size_t bufEndIndex = (wavePostTc + tcSample <= bufLength ? wavePostTc + tcSample : bufLength);
    bufEndIndex = int(tcSample + wavePostTc if tcSample + wavePostTc <= bufLength else bufLength)
    bufWaveLength = bufEndIndex - bufStartIndex
    waveStartIndex = int(0 if tcSample >= waveTcSample else waveTcSample - tcSample)

    if kwargs.get('window', True) and tcSample >= waveTcSample:
        # smoothly turn on waveform
        window = scipy.signal.tukey(bufWaveLength)
        window[int(0.5*bufWaveLength):] = 1.
    else:
        window = 1
    h_td = np.zeros(bufLength, dtype=complex)
    h_td[bufStartIndex:bufEndIndex] = window*hplus.data.data[waveStartIndex:waveStartIndex+bufWaveLength] -\
                                      1j*window*hcross.data.data[waveStartIndex:waveStartIndex+bufWaveLength]
    return h_td

NR_PATH = '/Users/maxisi/lscsoft/src/lvcnr-lfs/SXS/SXS_BBH_0305_Res6.h5'
def get_signal_dict(time_dict, lal=True, **kwargs):
    # get antenna patterns and trigger times
    tgps_dict = kwargs.pop('tgps_dict', None)
    ap_dict = kwargs.pop('ap_dict', None)
    if not tgps_dict or not ap_dict:
        ra, dec, psi = [kwargs.pop(k) for k in ['ra', 'dec', 'psi']]
        ifos = time_dict.keys()
        tgeo = kwargs.pop('tgps_geocent')
        tgps_dict, ap_dict = get_tgps_aps(tgeo, ra, dec, psi, ifos)
        
    # get complex strain
    time = list(time_dict.values())[0]
    delta_t = time[1] - time[0]
    approx = kwargs.pop('approx')
    if lal is True:
        hp, hc = generate_lal_hphc(approx,
                                   dt=delta_t, **kwargs)
    else:
        ell, mtot_msun, dist_mpc, inclination, phi_ref, modes = kwargs.pop('ell'), kwargs.pop('mtot_msun'), kwargs.pop('dist_mpc'), kwargs.pop('inclination'), kwargs.pop('phi_ref'), kwargs.pop('modes')
        strain_tmp, time_tmp = GetComplexStrainElls(ell, mtot_msun, dist_mpc, inclination, phi_ref, modes)
        hp_tmp = np.real(strain_tmp)
        hc_tmp = -np.imag(strain_tmp)
        #hp = interp1d(time_tmp, hp_tmp, bounds_error=False, fill_value=0.0, kind='cubic')
        #hc = interp1d(time_tmp, hc_tmp, bounds_error=False, fill_value=0.0, kind='cubic')
        hp = splrep(time_tmp, hp_tmp)
        hc = splrep(time_tmp, hc_tmp)

        #amp_tmp = np.abs(strain_tmp)
        #phase_tmp = np.angle(strain_tmp)
        #amp = interp1d(time_tmp, amp_tmp, bounds_error=False, fill_value=0.0) 
        #phase = interp1d(time_tmp, phase_tmp, bounds_error=False, fill_value=0.0) 
    # project signal onto detector
    raw_signal_dict = {}
    for ifo, time in time_dict.items():
        if lal is True:
            h = generate_lal_waveform(hplus=hp, hcross=hc, times=time,
                                      triggertime=tgps_dict[ifo],
                                      manual_epoch=kwargs.get('manual_epoch'))
        else:
            #h = hp(time-tgps_dict[ifo]) - 1j*hc(time-tgps_dict[ifo])
            t_tmp = time-tgps_dict[ifo]
            select = (t_tmp<=time_tmp[-1])*(t_tmp>=time_tmp[0])
            h = np.array([0.+0.j]*time.size)
            h[select] = splev(t_tmp[select], hp) - 1j*splev(t_tmp[select], hc)
            #amp_out, phase_out = amp(time-tgps_dict[ifo]), phase(time-tgps_dict[ifo])
            #h = amp_out*np.exp(phase_out)
        Fp, Fc = ap_dict[ifo]
        h_ifo = Fp*h.real - Fc*h.imag

        raw_signal_dict[ifo] = h_ifo
    return raw_signal_dict, tgps_dict, ap_dict


