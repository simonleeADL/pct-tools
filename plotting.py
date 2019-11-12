# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          K. Brueege <kai.bruegge@tu-dortmund.de>

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.stats import binned_statistic

from spectrum import CrabSpectrum

crab = CrabSpectrum()

def plot_crab_flux(bin_edges, ax=None):
    if not ax:
        ax = plt.gca()
    ax.plot(bin_edges, crab.flux(bin_edges) * bin_edges**2, 
        ls='-', lw=1, color='#3F3F5F', alpha=0.5, label='Crab Flux')
    return ax

def plot_sensitivity(rs, bin_edges, bin_center, color='black', ax=None, **kwargs):
    sensitivity = rs.sensitivity.values * (crab.flux(bin_center) * bin_center**2).to(u.erg / (u.s * u.cm**2))
    sensitivity_low = rs.sensitivity_low.values * (crab.flux(bin_center) * bin_center**2).to(u.erg / (u.s * u.cm**2))
    sensitivity_high = rs.sensitivity_high.values * (crab.flux(bin_center) * bin_center**2).to(u.erg / (u.s * u.cm**2))
    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    yerr = [np.abs(sensitivity - sensitivity_low).value, np.abs(sensitivity - sensitivity_high).value]

    if not ax:
        ax = plt.gca()
    ax.errorbar(bin_center.value, 
                sensitivity.value, 
                xerr=xerr, yerr=yerr, 
                linestyle='', ecolor=color, 
                **kwargs)
    return ax

def plot_ref_sens(ax=None):
	if not ax:
		ax = plt.gca()

	ref = np.loadtxt('references/MAGIC-50h.txt')
	ax.plot(ref[:,0],ref[:,1],
        '--', label='MAGIC 50 h')

	ref = np.loadtxt('references/ASTRI-50h.txt')
	ax.plot(ref[:,0],ref[:,1],
        '--', label='ASTRI 50 h')

	ref = np.loadtxt('references/South-50h-SST.txt')
	ax.plot(10**ref[:,0],ref[:,1],
        '--', label='SST sub-system 50 h')

	return ax


def plot_percentile(x, y, t, bins, bin_centers, bin_widths, ax=None):
	if not ax:
		ax = plt.gca()

	b_68, bin_edges, binnumber = binned_statistic(
            x, y, 
            statistic=lambda y: np.percentile(y, 68), 
            bins=bins
            )

	counts, _, _ = binned_statistic(
            x, y, 
            statistic='count', 
            bins=bins
            )

	min_counts = 100

	ax.errorbar(
            bin_centers.value[counts > min_counts],
            b_68[counts > min_counts],
            xerr=bin_widths.value[counts > min_counts] / 2.0,
            linestyle='',
            label=f'Threshold {t}',
            )

	return ax

