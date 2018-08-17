#!/usr/bin/env python3
"""
convert_fits_to_csv.py

Convert (batch) SWS .fits files to csv.

Note: I think dframe['wave'] is duplicated (identically) for all dframes,
      may want to remove later.
Note: CSV dframes use a lot more space than FITS. Not sure what format we
      want the data in for the neural network...?
Note: Not sure yet what to do with the header.
"""

import glob
import numpy as np
# import pandas as pd

from helpers import fits_to_dataframe
# from ipdb import set_trace as st

# Location of fits/csv files.
fits_dir = '../data/fits/'
fits_files = np.sort(glob.glob(fits_dir + '*.fit'))
csv_dir = '../data/csv/'

# Maybe save the dframes and headers.
hold_dframes = []
hold_header = []

# Iterate over all .fits files in 'data/fits/':
for index, fname in enumerate(fits_files):

    # Pull out base filename (minus extension and directory).
    base_fname = fname.split('/')[-1].split('.fit')[0]

    try:
        dframe, header = fits_to_dataframe(fname)
    except Exception as e:
        raise(e)
    else:
        csv_fname = base_fname + '.csv'

    # Save dataframe to a csv file; may want to leave as FITS later?
    dframe.to_csv(csv_dir + csv_fname)

    # Maybe hold onto the dframes and headers...
    hold_dframes.append(dframe)
    hold_header.append(header)
