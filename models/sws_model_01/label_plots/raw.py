#!/usr/bin/env python3
"""
raw.py

Plot the raw, unprocessed spectra.
"""

import matplotlib
matplotlib.use('Agg')

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import os.path  # noqa: E402
from time import time  # noqa: E402

import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def load_model_results(file_path):
    """Returns the model results as an ndarray."""
    def determine_max_prob(row):
        """Returns the maximum group probability."""
        prob = row[3:].astype(float)
        nan_present = np.sum([np.isnan(x) for x in prob])
        if nan_present:
            return np.nan, np.nan
        else:
            return np.nanmax(prob), np.nanargmax(prob)

    def determine_classifier(row):
        """Determine what group this result belongs to."""
        prob = row[3:].astype(float)

        if np.amax(prob) >= 0.80:
            return np.argmax(prob)
        else:
            return None

    def sort_list(sublist, sublist_index=None):
        """Sort the group lists by probability."""
        group_maximums = []

        # A bit slow compared to vector operations, but no chance of error.
        # Iterate over rows in each group.
        for row in sublist:
            max_prob, max_prob_arg = determine_max_prob(row)

            # Sanity check, maximum should match group position.
            if sublist_index is not None:
                assert sublist_index == max_prob_arg

            group_maximums.append(max_prob)

        # Order by highest probability.
        group_maximums = np.array(group_maximums)
        arg_sort = np.argsort(group_maximums)
        # sorted_group_maximums = group_maximums[arg_sort]

        # Sort sublist.
        sublist = np.array(sublist)
        sorted_sublist = list(sublist[arg_sort])[::-1]

        # Reindex the sublist.
        sort_arr = np.array(sorted_sublist)
        current_id = sort_arr.T[0]
        new_id = np.arange(len(current_id))
        new_id_str = [str(x).zfill(5) for x in new_id]
        sort_arr.T[0] = new_id_str

        return sort_arr.tolist()

    def parse_to_dict(data_in):
        """Create a dictionary of the results."""
        temp_lists = [[] for i in range(5)]
        uncertain_list = []
        group_dict = {}

        for row in data_in:
            # Determine what group this row belongs to.
            classifier = determine_classifier(row)
            # Sort the row into the list for just this group/classifier.
            if classifier is not None:
                temp_lists[classifier].append(row)
            else:
                uncertain_list.append(row)

        # Iterate over each group (0-4).
        for sublist_index, sublist in enumerate(temp_lists):
            # Sort the sublists by confidence.
            sorted_sublist = sort_list(sublist, sublist_index)
            # Store in dict.
            group_dict[sublist_index] = sorted_sublist

        # Iterate over uncertain_list.
        sorted_uncertain_list = sort_list(uncertain_list)
        group_dict['uncertain'] = sorted_uncertain_list

        return group_dict

    # Sort the model results into a dictionary, one key per group,
    # each sublist sorted by confidence (higher probability).
    data_in = np.loadtxt(file_path, delimiter=',', dtype='str')
    model_results = parse_to_dict(data_in)

    return model_results


def plot_row(input_tuple):
    """Create a PDF file of a row in the model results."""

    def plot_prediction(wave, flux, fluxerr, model_prediction,
                        output_directory, savename, file_path,
                        verbose=False, overwrite=False):

        if overwrite is False:
            if os.path.isfile(output_directory + savename):
                return

        # Plot relevant quantities.
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        # gs.update(wspace=0.025, hspace=0.00)  # spacing between axes.
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        # Axes 1: spectrum plot
        ax0.plot(wave, flux)
        ax0.axhline(y=0, color='red', ls='--', lw=2)
        ax0.set_title(file_path + '\np=' + str(np.nanmax(model_prediction)))

        # Axes 2: model prediction
        # ax1.bar()
        classifiers = ['Naked', 'Dusty', 'Warm dust', 'Cool dust', 'Very red']
        sns.barplot(x=classifiers, y=model_prediction, ax=ax1)

        # Save PDF.
        plt.savefig(output_directory + savename, format='pdf',
                    bbox_inches='tight')
        plt.close()
        fig.clear()

        if verbose:
            print('Saved: ', output_directory + savename)

        return

    # Necessary setup.
    row, output_directory = input_tuple
    row_index = row[0]
    data_dir = '../../../data/cassis/spectra/'

    # Identify aorkey, pickled file.
    aorkey = row[1]
    pickled_item = data_dir + aorkey + '.pkl'
    file_path = pickled_item.split('/cassis/')[-1]

    # Read in pickle.
    spectrum = pd.read_pickle(pickled_item)

    # Extract relevant quantities.
    wave = spectrum['wavelength'].values
    flux = spectrum['flux'].values
    fluxerr = spectrum['error (RMS+SYS)'].values
    model_prediction = np.array(row[3:]).astype(float)

    savename = 'plot_' + str(row_index).zfill(5) + '_' + aorkey + '.pdf'
    plot_prediction(wave, flux, fluxerr, model_prediction,
                    output_directory, savename, file_path,
                    verbose=False, overwrite=True)

    return


def main():

    results = load_model_results('../results.txt')

    # Iterate over all groups, save one PDF for group, sorted by confidence.
    for index, key in enumerate(results):
        if index in [0, 1, 2, 3, 4]:
            continue

        print(' ****  Group: ', key, ' **** ')

        # Extract results for a single group.
        group_results = results[key]
        group_len = len(group_results)

        # Where to save the PDFs.
        output_directory = 'plots_' + str(key) + '/'
        in_tuple = zip(group_results, [output_directory] * group_len)

        # Multiprocessing!
        max_workers = 12
        print('Max workers: ', max_workers)
        start = time()
        pool = ProcessPoolExecutor(max_workers=max_workers)
        results = list(pool.map(plot_row, in_tuple))
        # results = list(pool.map(run_NN, search_map))
        end = time()
        print('Took %.3f seconds' % (end - start))


if __name__ == '__main__':
    main()
