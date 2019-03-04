
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

# ### Subset 1: all data included (trimmed)

# In[5]:


features, labels = helpers.load_data(base_dir=base_dir, metadata=metadata,
                                     only_ok_data=False, clean=False, verbose=False,
                                     cut_28micron=False)


# In[6]:


print(features.shape)
print(labels.shape)


# ### Subset 2: exclude group 7 (trimmed)

# In[7]:


features_clean, labels_clean =     helpers.load_data(base_dir=base_dir, metadata=metadata,
                      only_ok_data=False, clean=True, verbose=False,
                      cut_28micron=False)


# In[8]:


print(features_clean.shape)
print(labels_clean.shape)


# ### Subset 3: exclude group 7, uncertain data (trimmed)

# In[9]:


features_certain, labels_certain =     helpers.load_data(base_dir=base_dir, metadata=metadata,
                      only_ok_data=True, clean=True, verbose=False,
                      cut_28micron=False, remove_group=6)


# In[10]:


print(features_certain.shape)
print(labels_certain.shape)


# In[11]:


np.unique(labels_certain)


# # Testing l2norms

# In[69]:


def neural(features, labels, test_size=0.3, l2norm=0.01):

    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=test_size, random_state = 42)

    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm), input_dim=303))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(5, activation='softmax'))

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


# In[70]:


# for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
#     model, test_size, accuracy = neural(features, labels, l2norm=l2norm)


# In[71]:


# for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
#     model, test_size, accuracy = neural(features_clean, labels_clean, l2norm=l2norm)


# In[72]:


# for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
#     model, test_size, accuracy = neural(features_certain, labels_certain, l2norm=l2norm)


# In[73]:


# for l2norm in (0.001, 0.0001, 0.00001, 0.000001):
#     model, test_size, accuracy = neural(features_certain, labels_certain, l2norm=l2norm)


# ***

# # Testing training size vs. accuracy

# Model:

# In[29]:


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
    model.add(keras.layers.Dense(5, activation='softmax'))

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

    return test_size, accuracy, model


# In[24]:


def run_networks(search_map):
    # Run the networks in parallel.
    start = time()
    pool = ProcessPoolExecutor(max_workers=7)
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

# In[25]:


# Values of test_size to probe.
# search_space = np.arange(0.14, 0.60, 0.02)
search_space = np.arange(0.32, 0.40, 0.04)
print('Size of test set considered: ', search_space)

# Number of iterations for each test_size value.
# n_iterations = 20
n_iterations = 4

# Create a vector to iterate over.
rx = np.array([search_space] * n_iterations).T
search_space_full = rx.flatten()

print('Number of iterations per test_size: ', n_iterations)
print('Total number of NN iterations required: ', n_iterations * len(search_space))


# In[26]:


# Wrap up tuple inputs for running in parallel.
# search_map = [(features, labels, x) for x in search_space_full]
# search_map_clean = [(features_clean, labels_clean, x) for x in search_space_full]
search_map_certain = [(features_certain, labels_certain, x) for x in search_space_full]


# In[27]:


# run_matrix = run_networks(search_map)
# run_matrix_clean = run_networks(search_map_clean)
run_matrix_certain = run_networks(search_map_certain)


# In[28]:


plot_results(run_matrix_certain)


# # Results - trimmed wavelength arrays

# ## Full set:

# In[14]:


plot_results(run_matrix)


# ## Clean set:

# In[15]:


plot_results(run_matrix_clean)


# ## Certain set: (< 28 µm only)

# In[25]:


plot_results(run_matrix_certain)


# ## Certain set: (full wavelength array)

# In[37]:


plot_results(run_matrix_certain)


# In[45]:


x, y = run_matrix_certain.T
z = np.polyfit(x, y, 4)
p = np.poly1d(z)


# In[47]:


xp = np.linspace(x[0], x[-1], 100)
_ = plt.plot(x, y, '.', xp, p(xp), '-', lw=2)


# # Remove group 6! only six!

# ## Certain set: (full wavelength array)

# In[31]:


plot_results(run_matrix_certain)


# In[32]:


x, y = run_matrix_certain.T
z = np.polyfit(x, y, 4)
p = np.poly1d(z)


# In[33]:


xp = np.linspace(x[0], x[-1], 100)
_ = plt.plot(x, y, '.', xp, p(xp), '-', lw=2)


# ***

# Based on the above, probably need to do more data preprocessing:
# - e.g., remove untrustworthy data

# # Run a single time, save to file.

# In[30]:


# Input for NN:
in_tuple = (features_certain, labels_certain, 0.35)

# Run and retrieve model.
test_size, accuracy, model = run_NN(in_tuple)
print(test_size, accuracy)


# In[34]:


save_path = '../models/sws_model_01.h5'
model.save(save_path)


# In[38]:


save_path_text = '../models/sws_model_01_features.txt'
np.savetxt(save_path_text, features_certain.T)


# In[35]:


features_certain.shape


# In[36]:


labels_certain.shape


# In[39]:


model.input


# In[ ]:


model. 

