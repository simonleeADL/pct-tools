# Authors: S. Einecke <sabrina.einecke@adelaide.edu.au>
#          K. Brueege <kai.bruegge@tu-dortmund.de>

import click
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import astropy.units as u

from utilities import read_data, make_energy_bins, add_theta
from plotting import plot_percentile

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--thresholds', default=0.0, multiple=True)
@click.option('-b', '--bins_number', default=15)
@click.option('--auto_energy', is_flag=True, default=False)
@click.option('--auto_limits', is_flag=True, default=False)

def main(input_file, output, thresholds, bins_number, auto_energy, auto_limits):
    
    df = read_data(input_file)
    source_az = df['true_source_az'][0]*u.rad
    df = add_theta(df,source_az=source_az.to(u.deg))
    
    if not auto_energy:
        e_min = 0.08; e_max = 300.0
    else:
        e_min = df['energy_range_min'][0]
        e_max = df['energy_range_max'][0]

    if not thresholds:
        thresholds = [0.0]

    bins, bin_centers, bin_widths = make_energy_bins(
        e_min=e_min * u.TeV, 
        e_max=e_max * u.TeV, 
        bins=bins_number,
        centering='log')

    ax = None

    for t in thresholds:

        e_true = df[df.gamma_score_mean > t].mc_energy.values
        e_reco = df[df.gamma_score_mean > t].energy_mean.values

        resolution = np.abs(e_reco - e_true) / e_true

        ax = plot_percentile(e_reco, resolution, t, bins, bin_centers, bin_widths, ax=ax)

    ax.plot([10**0,10**2.47], [0.1,0.1], '--', 
        color='silver', label='SST sub-system')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\mathrm{Reconstructed}\ \mathrm{Energy}\ /\  \mathrm{TeV}$')
    ax.set_ylabel('$\Delta E\ /\ E\ (68\% \ \mathrm{containment})$')
    if not auto_limits:
        ax.set_xlim([0.5,300])
        ax.set_ylim([0,0.5])
    ax.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
