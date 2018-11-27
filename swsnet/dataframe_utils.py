#!/usr/bin/env python3
"""
dataframe_utils.py

Utilities for pd.DataFrame manipulation for ipy notebooks.
"""

import errno
import os

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
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
    # specerr = spectrum['spec_error']
    # normerr = spectrum['norm_error']
    # fluxerr = specerr + normerr
    fluxerr = spectrum['uncertainty']

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
    ax.plot(spectrum_dict['wave'], spectrum_dict['flux'], **kwargs)
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
        spectrum_dict = read_spectrum('../../' + file_path)
        metadata_dict = read_metadata(dataframe, index)

        # Save spectrum as a PDF.
        plot_spectrum(spectrum_dict, metadata_dict,
                      save_dir=save_dir, **kwargs)

    return


def convert_fits_to_pickle(path, verify_pickle=False, verbose=False):
    """Full conversion from ISO-SWS <filename.fits to <filename>.pkl,
    which contains a pd.DataFrame.

    Args:
        path (str): Path to <filename>.fits file (of an ISO-SWS observation).
        verify_pickle (bool): Confirm the pickle was succesful created; does so
            by comparing the pd.DataFrame before and after writing the pickle.

    Returns:
        True if successful.

    Note:
        DataFrame can be retrieved from the pickle by, e.g.,
        df = pd.read_pickle(pickle_path).
    """

    if verbose:
        print('Pickling: ', path)

    # Convert .fits file to pandas DataFrame, header.Header object.
    try:
        df, header = isosws_fits_to_dataframe(path)
    except Exception as e:
        raise(e)

    # Determine the pickle_path to save to. Being explicit here to
    # 'pickle_path' is clear.
    base_filename = path.replace('.fit', '.pkl').split('/')[-1]

    # Save the dataframe to a pickle.
    pickle_path = 'spectra/' + base_filename
    df.to_pickle(pickle_path)

    if verbose:
        print('...saved: ', pickle_path)

    # Test dataframes for equality before/after pickling if
    # verify_pickle == True.
    if verify_pickle:
        tmp_df = pd.read_pickle(pickle_path)
        if df.equals(tmp_df):
            if verbose:
                print()
                print('DataFrame integrity verified -- pickling went OK!')
                print()
        else:
            raise ValueError('Dataframes not equal before/after pickling!')

    return pickle_path


def isosws_fits_to_dataframe(path, test_for_monotonicity=True):
    """Take an ISO-SWS .fits file, return a pandas DataFrame containing the
    data (with labels) and astropy header.

    Args:
        path (str): Path of the .fits file (assumed to be an ISO-SWS
            observation file).
        test_for_monotonicity (bool, optional): Check that the wavelength
            grid is monotinically increasing.

    Returns:
        df (pd.DataFrame): Pandas dataframe with appropriate labels
            (wavelength, flux, etc.).
        header (astropy.io.fits.header.Header): Information about observation
            from telescope.

    Note:
        Header can be manipulated with, e.g., header.totextfile(some_path).
        See http://docs.astropy.org/en/stable/io/fits/api/headers.html.
    """

    def monotonically_increasing(array):
        """Test if a list has monotonically increasing elements.
        Thank you stack overflow."""
        return all(x < y for x, y in zip(array, array[1:]))

    # Read in .fits file.
    hdu = fits.open(path)

    # Retrieve the header object.
    header = hdu[0].header

    # Extract column labels/descriptions from header.
    # Can't do this because the header is not well-defined. That's OK,
    # hard-coded the new column names below.

    # Convert data to pandas DataFrame.
    dtable = Table(hdu[0].data)
    df = dtable.to_pandas()

    # Convert the nondescriptive column labels (e.g., 'col01def', 'col02def')
    # to descriptive labels.
    old_keys = list(df.keys())
    new_keys = ['wavelength', 'flux', 'uncertainty']
    mydict = dict(zip(old_keys, new_keys))
    df = df.rename(columns=mydict)  # Renamed DataFrame columns here.

    monotonicity_msg = 'Wavelength array not monotonically increasing!'
    if test_for_monotonicity:
        if not monotonically_increasing(df['wavelength']):
            raise ValueError(monotonicity_msg, path)

    return df, header


def create_swsmeta_dataframe(good_tdts=None):
    """Create a dataframe that contains the metadata for the ISO-SWS Atlas."""

    def simbad_results():
        """Create a dictionary of the SIMBAD object type query results."""
        simbad_results = np.loadtxt('isosws_misc/simbad_type.csv',
                                    delimiter=';', dtype=str)
        simbad_dict = dict(simbad_results)
        return simbad_dict

    def sexagesimal_to_degree(tupe):
        """Convert from hour:minute:second to degrees."""
        sex_str = tupe[0] + ' ' + tupe[1]
        c = SkyCoord(sex_str, unit=(u.hourangle, u.deg))
        return c.ra.deg, c.dec.deg

    def transform_ra_dec_into_degrees(df):
        """Perform full ra, dec conversion to degrees."""
        ra = []
        dec = []
        for index, value in enumerate(zip(df['ra'], df['dec'])):
            ra_deg, dec_deg = sexagesimal_to_degree(value)
            ra.append(ra_deg)
            dec.append(dec_deg)
        df = df.assign(ra=ra)
        df = df.assign(dec=dec)
        return df

    # Read in the metadata
    # meta_filename = 'isosws_misc/kraemer_class.csv'
    meta_filename = 'isosws_misc/kraemer_class_fixed.csv'
    swsmeta = np.loadtxt(meta_filename, delimiter=';', dtype=str)
    df = pd.DataFrame(swsmeta[1:], columns=swsmeta[0])

    # Add a column for the pickle paths (dataframes with wave, flux, etc).
    pickle_paths = ['spectra/' + x.zfill(8) + '_irs.pkl' for x in df['tdt']]
    df = df.assign(file_path=pickle_paths)

    # Add a column for SIMBAD type, need to query 'simbad_type.csv' for this.
    # Not in order naturally...
    object_names = df['object_name']
    object_type_dict = simbad_results()
    object_types = [object_type_dict.get(key, "empty") for key in object_names]
    df = df.assign(object_type=object_types)

    # Transform ra and dec into degrees.
    df = transform_ra_dec_into_degrees(df)

    # Remove rows of objects not pickled (typically due to a data error).
    bool_list = []
    for path in df['file_path']:
        if os.path.isfile(path):
            bool_list.append(True)
        else:
            bool_list.append(False)

    df = df.assign(data_ok=bool_list)
    df = df.query('data_ok == True')
    df.reset_index(drop=True, inplace=True)

    # SORT BY TDT!
    df['tdt'] = df['tdt'].astype(int)
    df = df.sort_values(by=['tdt'], ascending=True)
    df = df.reset_index(drop=True)

    if good_tdts is not None:
        # Drop rows without data.
        drop_indices = []
        for row in df.itertuples(index=True, name='Pandas'):
            tdt = getattr(row, "tdt")
            if tdt not in good_tdts:
                drop_indices.append(row[0])

    final_df = df.drop(drop_indices)
    final_df = final_df.reset_index(drop=True)

    return final_df
