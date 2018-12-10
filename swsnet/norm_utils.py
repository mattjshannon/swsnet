#!/usr/bin/env python3
"""
norm_utils.py

Utilities for normalization in ipy notebooks (preprocessing).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mattpy.utils import smooth


def save_shift_figure(plot_dir, classifier, iso_filename, wave, flux,
                      smooth_flux, smooth_flux_shift, verbose=True):
    """Saves a .PDF figure of the raw and shifted-smoothed spectrum."""

    # Plot the raw spectrum and smoothed spectrum.
    plt.plot(wave, flux)
    plt.plot(wave, smooth_flux)
    plt.axhline(y=0, color='k', ls='--', lw=1)
    plt.axhline(y=np.nanmax(smooth_flux_shift), ls='--', lw=1,
                zorder=-10, color='k')

    # Save to disk.
    plt.title(iso_filename + ' - classifier: ' + classifier)
    savepath = plot_dir + iso_filename + '_A.pdf'
    plt.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.close()

    if verbose:
        print('Saved: ', savepath)

    return


def save_renorm_figure(plot_dir, classifier, iso_filename, renorm_wave,
                       renorm_flux_shift, smooth_flux_shift,
                       verbose=True):
    """Saves a .PDF figure of the renormalized spectrum."""

    # Plot the renormalized/shifted spectrum.
    plt.plot(renorm_wave, renorm_flux_shift)
    plt.plot(renorm_wave, smooth_flux_shift)
    plt.axhline(y=0, color='red', ls='--', lw=1)
    plt.axhline(y=1, ls='--', lw=1, zorder=-10, color='red')

    # Save to disk.
    plt.title(iso_filename + ' - classifier: ' + classifier)
    savepath = plot_dir + iso_filename + '_B.pdf'
    plt.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.close()

    if verbose:
        print('Saved: ', savepath)

    return


def read_spectrum_alt(file_path):
    """Returns an ISO spectrum (wave, flux, etc.) from a pickle."""
    try:
        spectrum = pd.read_pickle(file_path)
    except OSError as e:
        raise e

    wave = spectrum['wavelength']
    flux = spectrum['flux']

    try:
        fluxerr = spectrum['uncertainty']
    except Exception:
        pass

    try:
        fluxerr = spectrum['error (RMS+SYS)']
    except Exception:
        pass

    return wave, flux, fluxerr, spectrum


def smooth_spectrum(flux, **kwargs):
    """Returns a shifted, smoothed spectrum, ready for normalization."""
    spec_min = np.nanmin(flux)
    spec_max = np.nanmax(flux)
    # print(spec_min, spec_max)

    # Smooth it to find the general low/high points.
    smooth_flux = smooth(flux.values, **kwargs)

    # Smooth it to find the general low/high points, and shift to zero.
    smooth_flux_shift = smooth(flux.values - spec_min, **kwargs)

    # Upper normalization factor (s.t. the maximum of the continuum is 1.0).
    norm_factor = np.nanmax(smooth_flux_shift)

    return spec_min, spec_max, smooth_flux, smooth_flux_shift, norm_factor


def normalize_spectrum(file_path, classifier, plot=True, verbose=True):
    """Normalizes an ISO spectrum to span 0-1 (the main curvature)."""
    wave, flux, fluxerr, _ = read_spectrum_alt('../../' + file_path)

    # Shift spectrum and get normalization factors.
    # Minimum should now=0.0.
    spec_min, spec_max, smooth_flux, smooth_flux_shift, norm_factor = \
        smooth_spectrum(flux, window_len=40)

    # Final renormalized quantities.
    renorm_wave = wave
    renorm_flux_shift = (flux - spec_min) / norm_factor

    # Plotting directory
    plot_dir = 'plots/'

    # Save file name.
    iso_filename = file_path.split('/')[-1].split('.pkl')[0]

    if plot:
        # Save a figure showing the initial smooth/shift.
        save_shift_figure(plot_dir, classifier, iso_filename,
                          wave, flux, smooth_flux, smooth_flux_shift,
                          verbose=False)

        # Save a figure of the final renormalized spectrum.
        save_renorm_figure(plot_dir, classifier, iso_filename,
                           renorm_wave, renorm_flux_shift,
                           smooth_flux_shift / norm_factor,
                           verbose=False)

    return spec_min, spec_max, norm_factor


def renormalize_spectrum(file_path, norm_factors, verbose=True,
                         output_dir='../spectra_normalized/'):

    # Sanity check that the parameters are for this particular file.
    if file_path != norm_factors[0]:
        raise SystemExit('File paths do not match!')

    # Read the original pickled spectrum.
    full_file_path = '../../' + file_path
    wave, flux, fluxerr, spectrum = read_spectrum_alt(file_path=full_file_path)

    # Identify the scaling factors.
    _, spec_min, spec_max, norm_fac = norm_factors
    spec_min = float(spec_min)
    spec_max = float(spec_max)
    norm_fac = float(norm_fac)

    # Scale its flux using the norm factors.
    renorm_flux = (flux - spec_min) / norm_fac

    # Create a new pickle with the scaled spectrum (otherwise same structure).
    spectrum['flux'] = renorm_flux

    # Save new pickle.
    save_path = file_path.replace('.pkl', '_renorm.pkl')
    save_path = save_path.replace('spectra', 'spectra_normalized')
    full_save_path = '../../' + save_path
    spectrum.to_pickle(full_save_path)

    # Print 'Saved!' statement if verbose.
    if verbose:
        print('Saved: ', full_save_path)

    return save_path
