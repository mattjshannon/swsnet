#!/usr/bin/env python3
"""
test_helpers.py

Test whether helpers.py behaves as expected."""


import numpy as np
import os
import pandas as pd
import pytest

from swsnet.helpers import load_spectrum, load_data, fits_to_dataframe


@pytest.fixture(scope="module")
def pickle_file(tmpdir_factory):
    pickle_path = tmpdir_factory.mktemp("data").join("test.pkl")
    mock_df = pd.DataFrame({'wave': [1, 2, 3], 'flux': [84, 80, 75]})
    mock_df.to_pickle(pickle_path)
    return pickle_path


@pytest.fixture(scope="module")
def meta_file(tmpdir_factory, pickle_file):
    meta_path = tmpdir_factory.mktemp("").join("metadata.pkl")
    mock_dict = {
        'group': [1, 7, 7],
        'file_path': [str(pickle_file), str(pickle_file), str(pickle_file)]
        }

    mock_df = pd.DataFrame(mock_dict)
    mock_df.to_pickle(meta_path)
    return meta_path


def test_load_spectrum(pickle_file):
    """Test that we can load a spectrum from a pickle file."""
    with pytest.raises(IOError):
        load_spectrum('')

    flux = load_spectrum(pickle_file, normalize=False)
    assert np.all(flux.values == np.array([84, 80, 75]))

    flux = load_spectrum(pickle_file, normalize=True)
    assert flux.values[0] == 1


def test_load_data(meta_file):
    """Test that we can load a metadata pickle and its associated spectra."""
    with pytest.raises(Exception):
        load_data(base_dir='made_up', metadata='made_up')

    features, labels = load_data(metadata=meta_file, n_samples=3,
                                 clean=False, normalize=False)
    assert list(labels) == [0, 6, 6]
    assert features.shape == (3, 3)
    assert labels.shape[0] == 3

    features, labels = load_data(metadata=meta_file, n_samples=3,
                                 clean=True, normalize=False)
    assert list(labels) == [0]
    assert features.shape == (1, 3)
    assert labels.shape[0] == 1


def test_fits_to_dataframe():
    """Test that we can read a .FITS file and return a DataFrame."""
    fname = os.path.join(os.path.dirname(__file__), 'data',
                         '02400714_sws.fit')
    df, header = fits_to_dataframe(fname)
    df_dict = df.to_dict()
    dict_keys = list(df_dict.keys())

    assert df.shape == (48924, 4)
    assert dict_keys == ['wave', 'flux', 'spec_error', 'norm_error']
    assert len(header) == 52
