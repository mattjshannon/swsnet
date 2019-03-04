
# coding: utf-8

# # TF neural net with normalized ISO spectra

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from IPython.core.debugger import set_trace as st
from sklearn.model_selection import train_test_split
from time import time

# My modules
from swsnet import helpers

print(tf.__version__)


# ## Dataset: ISO-SWS (normalized, culled)

# In[2]:


# Needed directories
base_dir = '../data/isosws_atlas/'

# Pickles containing our spectra in the form of pandas dataframes:
spec_dir = base_dir + 'spectra_normalized/'
spec_files = np.sort(glob.glob(spec_dir + '*.pkl'))

# Metadata pickle (pd.dataframe). Note each entry contains a pointer to the corresponding spectrum pickle.
metadata = base_dir + 'metadata_step2_culled.pkl'


# #### Labels ('group'):
# 
# 1. Naked stars
# 2. Stars with dust
# 3. Warm, dusty objects
# 4. Cool, dusty objects
# 5. Very red objects
# 6. Continuum-free objects but having emission lines
# 7. Flux-free and/or fatally flawed spectra

# ### Subset 1: all data included

# In[3]:


features, labels = helpers.load_data(base_dir=base_dir, metadata=metadata,
                                     only_ok_data=False, clean=False, verbose=False)


# In[4]:


print(features.shape)
print(labels.shape)


# ### Subset 2: exclude group 7

# In[5]:


features_clean, labels_clean =     helpers.load_data(base_dir=base_dir, metadata=metadata,
                      only_ok_data=False, clean=True, verbose=False)


# In[6]:


print(features_clean.shape)
print(labels_clean.shape)


# ### Subset 3: exclude group 7, uncertain data

# In[7]:


features_certain, labels_certain =     helpers.load_data(base_dir=base_dir, metadata=metadata,
                      only_ok_data=True, clean=True, verbose=False)


# In[8]:


print(features_certain.shape)
print(labels_certain.shape)


# # Testing l2norms

# In[10]:


def neural(features, labels, test_size=0.3, l2norm=0.01):

    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=test_size, random_state = 42)

    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm), input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=5, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=keras.optimizers.Adam(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
    print("L2 norm, accuracy: ", l2norm, accuracy)
    
    return model, test_size, accuracy


# In[20]:


# for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
#     model, test_size, accuracy = neural(features, labels, l2norm=l2norm)


# In[17]:


# for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
#     model, test_size, accuracy = neural(features_clean, labels_clean, l2norm=l2norm)


# In[23]:


# for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
#     model, test_size, accuracy = neural(features_certain, labels_certain, l2norm=l2norm)


# In[24]:


# for l2norm in (0.001, 0.0001, 0.00001, 0.000001):
#     model, test_size, accuracy = neural(features_certain, labels_certain, l2norm=l2norm)


# ***

# # Testing training size vs. accuracy

# Model:

# In[9]:


def run_NN(input_tuple):
    """Run a Keras NN for the purpose of examining the effect of training set size.
    
    Args:
        features (ndarray): Array containing the spectra (fluxes).
        labels (ndarray): Array containing the group labels for the spectra.
        test_size (float): Fraction of test size relative to (test + training).
        
    Returns:
        test_size (float): Input test_size, just a sanity check!
        accuracy (float): Accuracy of this neural net when applied to the test set.
    """
    
    features, labels, test_size = input_tuple
    l2norm = 0.001
    
    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=test_size, random_state = 42)
    
    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm), input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=5, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=keras.optimizers.Adam(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
#     print("Test size, accuracy: ", test_size, accuracy)

    return test_size, accuracy


# In[11]:


def run_networks(search_map):
    # Run the networks in parallel.
    start = time()
    pool = ProcessPoolExecutor(max_workers=6)
    results = list(pool.map(run_NN, search_map))
    end = time()
    print('Took %.3f seconds' % (end - start))

    run_matrix = np.array(results)
    return run_matrix

def plot_results(run_matrix):
    # Examine results.
    plt.plot(run_matrix.T[0], run_matrix.T[1], 's', mfc='w', ms=5, mew=2, mec='r');
    plt.xlabel('Test size (fraction)');
    plt.ylabel('Test accuracy');
    plt.minorticks_on();
#     plt.xlim(left=0);  
    return


# Search space (training size):

# In[31]:


# Values of test_size to probe.
search_space = np.arange(0.14, 0.60, 0.02)
# search_space = np.arange(0.14, 0.60, 0.04)
print('Size of test set considered: ', search_space)

# Number of iterations for each test_size value.
n_iterations = 20
# n_iterations = 4

# Create a vector to iterate over.
rx = np.array([search_space] * n_iterations).T
search_space_full = rx.flatten()

print('Number of iterations per test_size: ', n_iterations)
print('Total number of NN iterations required: ', n_iterations * len(search_space))


# In[32]:


# Wrap up tuple inputs for running in parallel.
search_map = [(features, labels, x) for x in search_space_full]
search_map_clean = [(features_clean, labels_clean, x) for x in search_space_full]
search_map_certain = [(features_certain, labels_certain, x) for x in search_space_full]
# search_map_certaintrim = [(features_certaintrim, labels_certaintrim, x) for x in search_space_full]


# In[33]:


run_matrix = run_networks(search_map)
run_matrix_clean = run_networks(search_map_clean)
run_matrix_certain = run_networks(search_map_certain)
# run_matrix_certaintrim = run_networks(search_map_certaintrim)


# # Results - full wavelength arrays

# ## Full set:

# In[35]:


plot_results(run_matrix)


# ## Clean set:

# In[36]:


plot_results(run_matrix_clean)


# ## Certain set:

# In[37]:


plot_results(run_matrix_certain)


# ***

# Based on the above, probably need to do more data preprocessing:
# - e.g., remove untrustworthy data

# In[21]:


# save_path = '../models/nn_sorted_normalized_culled.h5'


# In[22]:


# model.save(save_path)

