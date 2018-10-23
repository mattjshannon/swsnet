
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


# ## Dataset: ISO-SWS (normalized)

# In[2]:


# Needed directories
base_dir = '../data/isosws_atlas/'

# Pickles containing our spectra in the form of pandas dataframes:
spec_dir = base_dir + 'spectra_normalized/'
spec_files = np.sort(glob.glob(spec_dir + '*.pkl'))

# Metadata pickle (pd.dataframe). Note each entry contains a pointer to the corresponding spectrum pickle.
metadata = base_dir + 'metadata_normalized.pkl'


# #### Labels ('group'):
# 
# 1. Naked stars
# 2. Stars with dust
# 3. Warm, dusty objects
# 4. Cool, dusty objects
# 5. Very red objects
# 6. Continuum-free objects but having emission lines
# 7. Flux-free and/or fatally flawed spectra

# N.B., these are shifted down by 1 in the labels (to span 0-6) for the model.

# ### Subset 1: all data included

# In[3]:


features, labels = helpers.load_data(base_dir=base_dir, metadata=metadata, clean=False, verbose=False)


# In[4]:


print(features.shape)
print(labels.shape)


# ### Subset 2: exclude group=7 data

# In[5]:


features_clean, labels_clean = helpers.load_data(base_dir=base_dir, metadata=metadata, clean=True, verbose=False)


# In[6]:


print(features_clean.shape)
print(labels_clean.shape)


# # The model itself

# In[9]:


def neural(features, labels, l2norm=0.01):

    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=0.3, random_state = 42)

    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm), input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l2norm)))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=keras.optimizers.Adam(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
    print("L2 norm, accuracy: ", l2norm, accuracy)
    
    return model, accuracy


# In[13]:


for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
    model, accuracy = neural(features, labels, l2norm=l2norm)


# In[14]:


for l2norm in (0.1, 0.01, 0.001, 0.0001, 0.00001):
    model, accuracy = neural(features_clean, labels_clean, l2norm=l2norm)


# Based on the above, probably need to do more data preprocessing:
# - e.g., remove untrustworthy data

# In[21]:


save_path = '../models/neural_net.h5'


# In[22]:


# model.save(save_path)

