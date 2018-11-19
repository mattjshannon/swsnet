#!/usr/bin/env python3
"""
helpers.py

Helper functions: e..g, read an SWS fits file, reformat for our uses.
"""

import numpy as np
import pandas as pd

from astropy.io import fits


def load_spectrum(path, normalize=True):
    """Extract the normalized flux vector from a pickled dataframe.

    Args:
        path (str): Path to the pickle file.
        normalize (bool): Whether to normalize the spectrum to its nanmax.

    Returns:
        flux (ndarray): The flux array.
    """

    try:
        dataframe = pd.read_pickle(path)
    except OSError as e:
        raise e

    flux = dataframe['flux']
    if normalize:
        flux = flux / np.nanmax(flux)

    return flux


def load_data(base_dir='', metadata='metadata.pkl', clean=False,
              only_ok_data=False, verbose=False, n_samples=359, **kwargs):
    """Load a pickled metadata file and extract features, labels.

    Args:
        base_dir (str): Path to the directory containing ISO data.
        metadata (str): Path to the metadata.pkl file.
        clean (bool): Whether to remove group=7 data from ISO.
        only_ok_data (bool): Whether to only include rows with data_ok == True.
        verbose (bool): Whether to plot/print diagnostics.

    Returns:
        features (ndarray): Array containing the spectra (just fluxes).
        labels (ndarray): Array containing the group classifier.
    """

    # Load the metadata pickle.
    try:
        meta = pd.read_pickle(metadata)
    except OSError as e:
        raise e

    if clean:
        meta = meta.query('group != "7"')
    if only_ok_data:
        meta = meta.query('data_ok == True')

    # Simple classifier first.
    labels = meta['group'].values.astype(int)
    np.unique(labels)

    # SHIFTING TO START AT ZERO!
    labels = labels - 1

    if verbose:
        # See how the labels are distributed.
        # plt.plot(labels, 'o');

        # Label type, length
        # print('VERBOSE.')
        print(type(labels[0]), len(labels))

    group_max = max([max(6, x) for x in labels])
    if group_max > 6:
        raise ValueError("Unexpected label: ", group_max)

    # Feature vector, knowing that each sample has a 359-point vector/spectrum.
    features = np.zeros((len(labels), n_samples))

    # Fill feature vector.
    index = 0

    # Fill the 'spectra' variable with the astronomical data.
    for row in meta.itertuples(index=True, name='Pandas'):
        try:
            flux = load_spectrum(base_dir + row.file_path, **kwargs)
        except OSError as e:
            raise e

        features[index] = flux
        index += 1

    return features, labels


def fits_to_dataframe(filename):
    """Convert an SWS FITS file to ascii.

    Args:
        filename (str): SWS .fits file.

    Returns:
        dframe (pd.DataFrame): wave, flux, error (spce), error (norm).
        header (fits.header): astropy header.

    Note:
        Column format of hdu[0].data.T:
        COMMENT Column 1 = wavelength (micron)
        COMMENT Column 2 = flux (Jy)
        COMMENT Column 3 = spectroscopic error (Jy)
        COMMENT Column 4 = normalization error (Jy)
    """

    # Read the FITS file, assign variables.
    try:
        hdu = fits.open(filename)
    except OSError as e:
        raise e

    assert isinstance(hdu, fits.hdu.hdulist.HDUList)

    try:
        header = hdu[0].header  # pylint: disable=no-member
        dat = hdu[0].data.T  # pylint: disable=no-member
    except Exception as e:
        raise e

    # Should be wave, flux, spec_error, norm_error
    if dat.shape[0] != 4:
        raise IndexError('Unexpected .fits file dimensions.')

    wave, flux, spec_error, norm_error = dat

    # Create a dictionary of useful measurements.
    hdu_dict = {'wave': wave, 'flux': flux,
                'spec_error': spec_error, 'norm_error': norm_error}

    # Make a pandas dataframe object out of our hdu dictionary.
    dframe = pd.DataFrame(hdu_dict)

    return dframe, header
