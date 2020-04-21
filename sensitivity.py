# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          K. Brueege <kai.bruegge@tu-dortmund.de>

import click
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from utilities import make_energy_bins, read_data, add_theta
from utilities import calc_relative_sensitivity
from plotting import plot_sensitivity, plot_crab_flux, plot_ref_sens

@click.command()
@click.argument('gamma_input', type=click.Path(exists=True))
@click.argument('proton_input', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--t_obs', type=float, default=50)
@click.option('--flux', default=True)
@click.option('--ref', default=True)
@click.option('-b', '--bins_number', default=15)
@click.option('--auto_energy', is_flag=True, default=False)

def main(gamma_input, proton_input, output, t_obs, flux, ref, bins_number, auto_energy):
    t_obs *= u.h

    gammas = read_data(gamma_input, weight=True, spectrum='crab', t_obs=t_obs)
    source_az = gammas['true_source_az'][0]*u.rad 
    protons = read_data(proton_input, weight=True, spectrum='proton', t_obs=t_obs)

    gammas = add_theta(gammas,source_az=source_az.to(u.deg))
    protons = add_theta(protons,source_az=source_az.to(u.deg))
    
    if not auto_energy:
        e_min = 0.08; e_max = 300.0
    else:
        e_min = gammas['energy_range_min'][0]
        e_max = gammas['energy_range_max'][0]

    bins, bin_centers, bin_widths = make_energy_bins(
        e_min=e_min * u.TeV, 
        e_max=e_max * u.TeV, 
        bins=bins_number,
        centering='log'
        )

    rel_sens = calc_relative_sensitivity(gammas, protons, bins, 
                                method='exact', alpha=0.2)

    ax = plot_sensitivity(rel_sens, 
                          bins, bin_centers,
                          label=f'This Analysis {t_obs:2.0f}')

    if flux:
        ax = plot_crab_flux(bins, ax)

    if ref:
        ax = plot_ref_sens(ax)

    ax.text(0.95, 0.95, 'Differential Sensitivity',
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1E-2, 10**(2.5)])
    ax.set_ylim([0.8E-13, 2E-10])
    ax.set_ylabel(r'$ E^2 \times \mathrm{Flux}\ \mathrm{Sensitivity} \ / \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-2}$)')
    ax.set_xlabel(r'$\mathrm{Reconstructed}\ \mathrm{Energy}\ E\ /\  \mathrm{TeV}$')
    ax.legend(loc='lower left')

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
