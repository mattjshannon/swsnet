#!/usr/bin/env python3
"""
overplot.py

Plot spectra for each possible SWS/CASSIS match.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ipdb import set_trace as st


def load_matches():
    """Returns the matches text file."""
    match_file = 'matches_within_11arcsec.txt'
    match_matrix = np.loadtxt(match_file, delimiter=',', dtype=str)
    return match_matrix


def plot_spectra(line_in, line_index):
    """Plot and save the spectra for a single line in the match matrix."""

    def load_cassis_spectrum(aor):
        """Returns a dict of wave, flux, fluxerr from a CASSIS pickle."""
        file_in = '../cassis/spectra/' + aor + '.pkl'
        dframe = pd.read_pickle(file_in)

        wave = dframe.wavelength.values
        flux = dframe.flux.values
        fluxerr = dframe['error (RMS+SYS)'].values

        spectral_dict = {
            'wave': wave,
            'flux': flux,
            'fluxerr': fluxerr
        }

        return spectral_dict

    def load_sws_spectrum(tdt):
        """Returns a dict of wave, flux, fluxerr from an SWS pickle."""
        file_in = '../isosws_atlas/spectra/' + tdt.zfill(8) + '_sws.pkl'
        dframe = pd.read_pickle(file_in)

        wave = dframe.wavelength.values
        flux = dframe.flux.values
        spec_error = dframe['spec_error']
        norm_error = dframe['norm_error']
        fluxerr = spec_error + norm_error

        spectral_dict = {
            'wave': wave,
            'flux': flux,
            'fluxerr': fluxerr
        }

        return spectral_dict

    # Define relevant quantities from input.
    aor, tdt, ang_sep, cassis_index, sws_index = line_in

    # Load spectral data.
    cassis = load_cassis_spectrum(aor)
    sws = load_sws_spectrum(tdt)

    # Make a plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cassis['wave'], cassis['flux'], label='cassis')
    ax.plot(sws['wave'], sws['flux'], label='sws')
    ax.legend(loc=0)

    title_str = 'AOR: ' + aor + ', TDT: ' + tdt
    ax.set_title(title_str)

    save_name = 'plot_' + str(line_index).zfill(4) + '_aor_' + \
        str(aor) + '_tdt_' + str(tdt) + '.pdf'

    fig.savefig('plots/' + save_name, bbox_inches='tight')
    plt.close()

    return


matches = load_matches()

for index, line in enumerate(matches):
    plot_spectra(line, index)
