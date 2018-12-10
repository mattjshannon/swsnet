#!/usr/bin/env python3
"""
raw.py

Plot the raw, unprocessed spectra.
"""

import glob

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ipdb import set_trace as st


def load_model_results(file_path):
    """Returns the model results as an ndarray."""
    file_in = np.loadtxt(file_path, delimiter=',', dtype='str')
    return file_in

def query_model_results(aorkey, results):
    """Returns the results for a given aorkey."""
    aorkey_list = results.T[1].tolist()

    # Ensure there's only one match for a given AORkey.
    assert aorkey_list.count(aorkey) == 1
    
    # Find result for specific AORkey.
    index_of_aorkey = aorkey_list.index(aorkey)
    results_row = results[index_of_aorkey]

    # Probabilities that spectrum belongs to a specific group:
    prediction = results_row[3:].astype(float)

    return prediction


results = load_model_results('../results.txt')
data_dir = '../../../data/cassis/spectra/'
pickle_list = np.sort(glob.glob(data_dir + '*.pkl'))

# Plot all dem pickles.
for index, pickled_item in enumerate(pickle_list):
    if index % 100 == 0:
        print(index, ' / ', len(pickle_list))

    """Make a PDF plot for each pickled spectrum."""
    spectrum = pd.read_pickle(pickled_item)
    file_path = pickled_item.split('/cassis/')[-1]
    aorkey = pickled_item.split('/')[-1].split('.pkl')[0]

    # Extract relevant quantities.
    wave = spectrum['wavelength'].values
    flux = spectrum['flux'].values
    fluxerr = spectrum['error (RMS+SYS)'].values
    model_prediction = query_model_results(aorkey, results)

    # Plot relevant quantities.
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    # gs.update(wspace=0.025, hspace=0.00)  # spacing between axes.
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # Axes 1: spectrum plot
    ax0.plot(wave, flux)
    ax0.axhline(y=0, color='red', ls='--', lw=2)
    ax0.set_title(file_path)

    # Axes 2: model prediction
    # ax1.bar()
    classifiers = ['Naked', 'Dusty', 'Warm dust', 'Cool dust', 'Very red']
    sns.barplot(x=classifiers, y=model_prediction, ax=ax1)

    # Save PDF.
    savename = pickled_item.split('/')[-1].split('.pkl')[0] + '.pdf'
    plt.savefig('plots/' + savename, format='pdf', bbox_inches='tight')
    plt.close()
    # print('Saved: ', 'plots/' + savename)

    # raise SystemExit