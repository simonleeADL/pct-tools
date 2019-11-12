# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          K. Brueege <kai.bruegge@tu-dortmund.de>

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

from scipy.optimize import brute, minimize_scalar

import fact.io

from spectrum import CosmicRaySpectrum, CrabSpectrum, MCSpectrum

crab = CrabSpectrum()
cosmic_proton = CosmicRaySpectrum()

def read_data(input_file, weight=False, spectrum=None, t_obs=50 * u.h):
    columns = ['gamma_score_mean', 'energy_mean', 
               'source_az_mean', 'source_alt_mean', #'altitude_raw', 'azimuth_raw', 
               'mc_energy']

    df_arr = fact.io.read_data(input_file, key='array_events')
    df_tel = fact.io.read_data(input_file, key='telescope_events')

    df = pd.merge(df_tel, df_arr, 
            on=['array_event_id', 'run_id'])
    df = df[columns].dropna()

    runs = fact.io.read_data(input_file, key='runs')
    mc_production = MCSpectrum.from_cta_runs(runs)

    if weight:
        if spectrum == 'crab':
            df['weight'] = mc_production.reweigh_to_other_spectrum(crab, 
                    df.mc_energy.values * u.TeV, t_assumed_obs=t_obs)
        elif spectrum == 'proton':
            df['weight'] = mc_production.reweigh_to_other_spectrum(cosmic_proton, 
                    df.mc_energy.values * u.TeV, t_assumed_obs=t_obs)

    return df

def calc_LiMa(n_on, n_off, alpha=0.2):
    scalar = np.isscalar(n_on)

    n_on = np.array(n_on, copy=False, ndmin=1)
    n_off = np.array(n_off, copy=False, ndmin=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_on = n_on / (n_on + n_off)
        p_off = n_off / (n_on + n_off)

        t1 = n_on * np.log(((1 + alpha) / alpha) * p_on)
        t2 = n_off * np.log((1 + alpha) * p_off)

        ts = (t1 + t2)
        significance = np.sqrt(ts * 2)

    significance[np.isnan(significance)] = 0
    significance[n_on < alpha * n_off] = 0

    if scalar:
        return significance[0]

    return significance


@u.quantity_input(energies=u.TeV, e_min=u.TeV, e_max=u.TeV)
def make_energy_bins(
        energies=None,
        e_min=None,
        e_max=None,
        bins=10,
        centering='linear',
):
    if energies is not None and len(energies) >= 2:
        e_min = min(energies)
        e_max = max(energies)

    unit = e_min.unit

    low = np.log10(e_min.value)
    high = np.log10(e_max.value)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * unit

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths


def add_theta(df, source_alt=70*u.deg, source_az=0*u.deg):

    source_az = Angle(source_az).wrap_at(180 * u.deg)
    source_alt = Angle(source_alt)

    az = Angle(df.source_az_mean.values, unit=u.rad).wrap_at(180 * u.deg)
    alt = Angle(df.source_alt_mean.values, unit=u.rad)

    df['theta'] = angular_separation(source_az, source_alt, az, alt).to(u.deg).value

    return df


def count_events_in_region(df, theta2=0.03, prediction_threshold=0.5):
    m = ((df.theta**2 <= theta2) & (df.gamma_score_mean >= prediction_threshold))
    return df[m].weight.sum(), m.sum()


def select_events_in_energy_range(signal, background, e_low, e_high, 
                                    use_true_energy=False):

    column = 'mc_energy' if use_true_energy else 'energy_mean'

    m = ((signal[column] > e_low) & (signal[column] < e_high))
    s = signal[m]

    m = ((background[column] > e_low) & (background[column] < e_high))
    b = background[m]

    return s, b


def scaling_factor(n_signal, n_background, t_signal, t_background, alpha=1, N=200):
    right_bound = 100

    def target(scaling_factor, n_signal, n_background, alpha=1, sigma=5):
        n_on = n_background * alpha + n_signal * scaling_factor
        n_off = n_background

        significance = calc_LiMa(n_on, n_off, alpha=alpha)
        return (5 - significance)**2

#     print(t_background, n_background, '---------', t_signal, n_signal)
    n_signal = np.random.poisson(t_signal, size=N) * n_signal / t_signal
    n_background = np.random.poisson(t_background, size=N) * n_background / t_background


    hs = []
    for signal, background in zip(n_signal, n_background):
        if background == 0:
            hs.append(np.nan)
        else:
            result = minimize_scalar(target, args=(signal, background, alpha), 
                                bounds=(0, right_bound), method='bounded').x
            if np.allclose(result, right_bound):
                result = np.nan
            hs.append(result)
    return np.nanpercentile(np.array(hs), (50, 5, 95))


def find_best_cuts(signal, background, alpha, regions=slice(0.0025, 0.08, 0.0025), 
                    thresholds=slice(0.05, 1, 0.05), method='simple'):

    def significance_target(cuts, signal, background, alpha):
        theta2, p_cut = cuts
        n_signal, t_signal = count_events_in_region(signal, theta2=theta2, 
                                                prediction_threshold=p_cut)

        if method == 'exact':
            n_background, t_background = count_events_in_region(background, 
                            theta2=theta2 / alpha, prediction_threshold=p_cut)

            if t_background < 10:
                #print(f'{cuts} not enough background')
                return 0

#         if t_background/alpha < 1:
#             print(f'{cuts} not enough background')
#             return 0

        if t_signal <= t_background * alpha + 10:
            #print('counts not large enough')
            return 0


        if t_signal <= t_background * alpha + 10:
            print('signal not large enough')
            return 0
        if n_signal*5 < n_background * 0.01:
            print('sys problem')
            return 0


        n_on = n_signal + alpha * n_background
        n_off = n_background
        return -calc_LiMa(n_on, n_off, alpha=alpha)

    result = brute(significance_target, ranges=[regions, thresholds], 
                            args=(signal, background, alpha), finish=None)
    
    print(result)
    
    return result


def calc_relative_sensitivity(signal, background, bin_edges, alpha=1, 
                                use_true_energy=False, method='exact'):
    relative_sensitivities = []
    thresholds = []
    thetas = []

    for e_low, e_high in (zip(bin_edges[:-1], bin_edges[1:])):
        s, b = select_events_in_energy_range(signal, background, e_low, e_high, 
                                            use_true_energy=use_true_energy)

        theta2, cut = find_best_cuts(s, b, alpha=alpha, method=method)

        n_signal, t_signal = count_events_in_region(s, theta2=theta2, 
                                                prediction_threshold=cut)
        
        if method == 'exact':
            n_background, t_background = count_events_in_region(b, 
                theta2=theta2 / alpha, prediction_threshold=cut)

        rs = scaling_factor(n_signal, n_background, t_signal, t_background, 
                            alpha=alpha)
        relative_sensitivities.append(rs)
        thresholds.append(cut)
        thetas.append(np.sqrt(theta2))

    m, l, h = np.array(relative_sensitivities).T
    d = {'sensitivity': m, 'sensitivity_low': l, 'sensitivity_high': h, 
        'threshold':thresholds, 'theta':thetas, 'e_min': bin_edges[:-1], 
        'e_max': bin_edges[1:]}
    return pd.DataFrame(d)







