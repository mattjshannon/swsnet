#!/usr/bin/env python3
"""
test_helpers.py

Test whether helpers.py behaves as expected."""
# pylint:disable=redefined-outer-name

import numpy as np
import pandas as pd
import pkg_resources
import pytest

from swsnet.helpers import load_spectrum, load_data, fits_to_dataframe


@pytest.fixture(scope="module")
def pickle_file(tmpdir_factory):
    """Fixture for a fake spectrum pickle."""
    pickle_path = tmpdir_factory.mktemp("data").join("test.pkl")
    mock_df = pd.DataFrame({'wave': [1, 2, 3], 'flux': [84, 80, 75]})
    mock_df.to_pickle(pickle_path)
    return pickle_path


@pytest.fixture(scope="module")
def pickle_file_long(tmpdir_factory):
    """Fixture for a fake spectrum pickle."""
    pickle_path = tmpdir_factory.mktemp("data").join("test.pkl")
    mock_df = pd.DataFrame({'wave': np.zeros(359), 'flux': np.zeros(359)})
    mock_df.to_pickle(pickle_path)
    return pickle_path


@pytest.fixture(scope="module")
def meta_file(tmpdir_factory, pickle_file):
    """Fixture for a fake metadata pickle."""
    meta_path = tmpdir_factory.mktemp("").join("metadata.pkl")
    mock_dict = {
        'group': [1, 7, 7],
        'file_path': [str(pickle_file)] * 3,
        'data_ok': [1, 1, 0]
    }
    mock_df = pd.DataFrame(mock_dict)
    mock_df.to_pickle(meta_path)
    return meta_path


@pytest.fixture(scope="module")
def meta_file2(tmpdir_factory, pickle_file):
    """Fixture for a fake metadata pickle, including invalid name."""
    meta_path2 = tmpdir_factory.mktemp("").join("metadata2.pkl")
    mock_dict2 = {
        'group': [1, 7, 7],
        'file_path': [str(pickle_file), str(pickle_file), 'fake.fakefile'],
        'data_ok': [1, 1, 0]
    }
    mock_df2 = pd.DataFrame(mock_dict2)
    mock_df2.to_pickle(meta_path2)
    return meta_path2


@pytest.fixture(scope="module")
def meta_file3(tmpdir_factory, pickle_file_long):
    """Fixture for a fake metadata pickle."""
    meta_path = tmpdir_factory.mktemp("").join("metadata3.pkl")
    mock_dict = {
        'group': [1, 7, 7],
        'file_path': [str(pickle_file_long)] * 3,
        'data_ok': [1, 1, 0]
    }
    mock_df = pd.DataFrame(mock_dict)
    mock_df.to_pickle(meta_path)
    return meta_path


@pytest.fixture(scope="module")
def meta_file_inf(tmpdir_factory, pickle_file):
    """Fixture for a fake metadata pickle, including NaN."""
    meta_path = tmpdir_factory.mktemp("").join("metadata_inf.pkl")
    mock_dict = {
        'group': [1, np.nan, 8],
        'file_path': [str(pickle_file)] * 3,
        'data_ok': [1, 1, 0]
    }
    mock_df = pd.DataFrame(mock_dict)
    mock_df.to_pickle(meta_path)
    return meta_path


def test_load_spectrum(pickle_file):
    """Test that we can load a spectrum from a pickle file."""
    with pytest.raises(OSError):
        load_spectrum('')

    # Load a pickle without normalizing the fluxes.
    flux = load_spectrum(pickle_file, normalize=False)
    assert np.all(flux.values == np.array([84, 80, 75]))

    # Load a pickle with noramlized fluxes.
    flux = load_spectrum(pickle_file, normalize=True)
    assert flux.values[0] == 1


def test_load_data(meta_file, meta_file2, meta_file3, meta_file_inf):
    """Test that we can load a metadata pickle and its associated spectra."""

    # File doesn't exist.
    with pytest.raises(OSError):
        load_data(base_dir='made_up', metadata='made_up')

    # Test reading a regular metadata.pkl.
    features, labels = load_data(metadata=meta_file, n_samples=3,
                                 clean=False, normalize=False)
    assert list(labels) == [0, 6, 6]
    assert features.shape == (3, 3)
    assert labels.shape[0] == 3

    # Test a metadata.pkl that points to a file that doesn't exist.
    with pytest.raises(OSError):
        features, labels = load_data(metadata=meta_file2, n_samples=3,
                                     clean=False,
                                     normalize=False)

    # Test a legit metadata.pkl with cleaning needed.
    features, labels = load_data(metadata=meta_file, n_samples=3,
                                 clean=True, normalize=False)
    assert list(labels) == [0]
    assert features.shape == (1, 3)
    assert labels.shape[0] == 1

    # Test a legit metadata.pkl with cleaning needed.
    features, labels = load_data(metadata=meta_file, n_samples=3, verbose=True,
                                 clean=True, only_ok_data=True,
                                 normalize=False)
    assert list(labels) == [0]
    assert features.shape == (1, 3)
    assert labels.shape[0] == 1

    # Labels outside of expected range for SWS "group" classifier.
    with pytest.raises(ValueError):
        _, _ = load_data(metadata=meta_file_inf, n_samples=3,
                         clean=False, normalize=False)

    # Try cutting the 28-micron zone.
    features, labels = load_data(metadata=meta_file, n_samples=3,
                                 clean=False, normalize=False,
                                 cut_28micron=True)
    assert list(labels) == [0, 6, 6]
    assert features.shape == (3, 3)
    assert labels.shape[0] == 3

    # Try cutting the 28-micron zone, long.
    features, labels = load_data(metadata=meta_file3, n_samples=359,
                                 clean=False, normalize=False,
                                 cut_28micron=True)
    assert features.shape == (3, 303)
    assert labels.shape[0] == 3

    # Try removing a specific group.
    features, labels = load_data(metadata=meta_file, n_samples=3,
                                 clean=False, normalize=False,
                                 remove_group=7)
    assert list(labels) == [0]
    assert features.shape == (1, 3)
    assert labels.shape[0] == 1


def test_fits_to_dataframe():
    """Test that we can read a .FITS file and return a DataFrame."""

    # First using real, well-formed data.
    test_fits = 'test_data/02400714_sws.fit'
    fname = pkg_resources.resource_filename('swsnet', test_fits)
    dataframe, header = fits_to_dataframe(fname)
    dataframe_dict = dataframe.to_dict()
    dict_keys = list(dataframe_dict.keys())

    assert dataframe.shape == (48924, 4)
    assert dict_keys == ['wave', 'flux', 'spec_error', 'norm_error']
    assert len(header) == 52

    # Next using a non-existent file.
    with pytest.raises(OSError):
        dataframe, header = fits_to_dataframe(fname + 'zzzz')

    # And now using real, ill-formed data.
    with pytest.raises(IndexError):
        test_fits = 'test_data/02400714_sws_err.fit'
        fname = pkg_resources.resource_filename('swsnet', test_fits)
        dataframe, header = fits_to_dataframe(fname)
