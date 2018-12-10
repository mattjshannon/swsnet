#!/usr/bin/env python3
"""
raw.py

Plot the raw, unprocessed spectra.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = '../../spectra/'
pickle_list = np.sort(glob.glob(data_dir + '*.pkl'))

# Plot all dem pickles.
for index, pickled_item in enumerate(pickle_list):
    if index % 100 == 0:
        print(index, ' / ', len(pickle_list))

    """Make a PDF plot for each pickled spectrum."""
    spectrum = pd.read_pickle(pickled_item)
    file_path = pickled_item.split('../')[-1]

    # Extract relevant quantities.
    wave = spectrum['wavelength'].values
    flux = spectrum['flux'].values
    fluxerr = spectrum['error (RMS+SYS)'].values

    # Plot relevant quantities.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wave, flux)
    ax.axhline(y=0, color='red', ls='--', lw=2)
    ax.set_title(file_path)

    # Save PDF.
    savename = pickled_item.split('/')[-1].split('.pkl')[0] + '.pdf'
    plt.savefig('plots/' + savename, format='pdf', bbox_inches='tight')
    print('Saved: ', 'plots/' + savename)
    plt.close()
