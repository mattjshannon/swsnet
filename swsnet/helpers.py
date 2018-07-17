#!/usr/bin/env python3
"""
helpers.py

Helper functions: e..g, read an SWS fits file, reformat for our uses.
"""

import pandas as pd
from astropy.io import fits
from ipdb import set_trace as st


def fits_to_dataframe(filename):
    """Convert an SWS FITS file to ascii.

    Args:
        filename (str): SWS .fits file.

    Returns:
        dframe (pd.DataFrame): wave, flux, error (spce), error (norm).
        header (fits.header): astropy header.
    """

    # Read the FITS file, assign variables.
    try:
        hdu = fits.open(filename)
    except Exception as e:
        raise(e)

    header = hdu[0].header
    dat = hdu[0].data.T

    try:
        wave, flux, spec_error, norm_error = dat
    except Exception as e:
        print('Unexpected .fits file dimensions.')
        raise(e)

    """
    Column format of hdu[0].data.T:
    COMMENT Column 1 = wavelength (micron)
    COMMENT Column 2 = flux (Jy)
    COMMENT Column 3 = spectroscopic error (Jy)
    COMMENT Column 4 = normalization error (Jy)
    """

    # Create a dictionary of useful measurements.
    hdu_dict = {'wave': wave, 'flux': flux,
                'spec_error': spec_error, 'norm_error': norm_error}

    # Make a pandas dataframe object out of our hdu dictionary.
    dframe = pd.DataFrame(hdu_dict)

    return dframe, header


def main():
    filename = '../data/fits/02400714_sws.fit'
    print('Testing process_fits_to_ascii for...')
    print(filename)
    print('Returning dframe, header.')
    dframe, header = process_fits_to_ascii(filename)

    st()
    return


if __name__ == '__main__':
    main()
