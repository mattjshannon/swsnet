#!/usr/bin/env python3
"""
dataframe_utils.py

Utilities for pd.DataFrame manipulation for ipy notebooks.
"""

import errno
import os

import matplotlib.pyplot as plt
import pandas as pd


def ensure_exists(path):
    """Ensure the path exists; if not, make the directory."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def read_spectrum(file_path):
    """Returns an ISO spectrum (wave, flux, etc.) from a pickle."""
    spectrum = pd.read_pickle(file_path)

    wave = spectrum['wavelength']
    flux = spectrum['flux']
    specerr = spectrum['spec_error']
    normerr = spectrum['norm_error']
    fluxerr = specerr + normerr

    spectrum_dict = {
        'wave': wave,
        'flux': flux,
        'fluxerr': fluxerr,
    }

    return spectrum_dict


def read_metadata(metadata_df, index):
    """Returns a dictionary of useful metadata."""
    file_path = metadata_df['file_path'].values[index]
    base_name_pkl = file_path.split('/')[-1]

    metadata_dict = {
        'base_name_pkl': base_name_pkl,
        'classifier': metadata_df['full_classifier'].values[index],
        'obj_type': metadata_df['object_type'].values[index],
        'obj_name': metadata_df['object_name'].values[index]
    }

    return metadata_dict


def plot_spectrum(spectrum_dict, metadata_dict, pdfpages=None,
                  save_dir='step5_cull_suspect/plots/',
                  verbose=False, **kwargs):
    """Save a PDF of a spectrum, needs a spectrum dict and metadata dict."""
    # Needed source properties.
    base_name_pkl = metadata_dict['base_name_pkl']
    classifier = metadata_dict['classifier']
    obj_type = metadata_dict['obj_type']
    obj_name = metadata_dict['obj_name']

    # Plot relevant quantities/labels.
    fig, ax = plt.subplots()
    ax.plot(spectrum_dict['wave'], spectrum_dict['flux'])
    ax.set_title(classifier + '   -   ' + base_name_pkl)
    ax.text(0.05, 0.95, obj_type, transform=ax.transAxes,
            color='red', ha='left')
    ax.text(0.95, 0.95, obj_name, transform=ax.transAxes,
            color='red', ha='right')

    # Save to PDF.
    ensure_exists(save_dir)
    savename = base_name_pkl.replace('.pkl', '.pdf')

    if pdfpages is not None:
        pdfpages.savefig(fig, bbox_inches='tight')
        plt.close()
    else:
        fig.savefig(save_dir + savename, format='pdf', bbox_inches='tight')
        plt.close()
        fig.clear()

    if verbose:
        print('Saved: ', save_dir + savename)

    return


def plot_dataframe(dataframe, save_dir, **kwargs):
    """Saves to PDF all spectra in a DataFrame."""

    # Iterate over the rows of the given input dataframe..
    for index, data_ok in enumerate(dataframe['data_ok']):
        if not data_ok:
            continue

        # Read in spectrum and metadata for given row.
        file_path = dataframe['file_path'].values[index]
        spectrum_dict = read_spectrum('../' + file_path)
        metadata_dict = read_metadata(dataframe, index)

        # Save spectrum as a PDF.
        plot_spectrum(spectrum_dict, metadata_dict,
                      save_dir=save_dir, **kwargs)

    return
