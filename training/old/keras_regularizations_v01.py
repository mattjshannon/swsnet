
# coding: utf-8

# # Testing keras regularizations

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


# In[2]:


# def run_NN(features, labels, test_size):
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
    
    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=test_size, random_state = 42)
    
    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callback, verbose=False)

#     # Summary
#     model.summary()

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])

    return test_size, accuracy


# ## Dataset: ISO-SWS

# In[3]:


# Needed directories
base_dir = '../data/isosws_atlas/'

# Pickles containing our spectra in the form of pandas dataframes:
spec_dir = base_dir + 'spectra/'
spec_files = np.sort(glob.glob(spec_dir + '*.pkl'))

# Metadata pickle (pd.dataframe). Note each entry contains a pointer to the corresponding spectrum pickle.
metadata = base_dir + 'metadata.pkl'


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

# In[4]:


features, labels = helpers.load_data(base_dir=base_dir, metadata=metadata, clean=False, verbose=False)


# In[5]:


print(features.shape)
print(labels.shape)


# ### Subset 2: exclude group=7 data

# In[6]:


features_clean, labels_clean = helpers.load_data(base_dir=base_dir, metadata=metadata, clean=True, verbose=False)


# In[7]:


print(features_clean.shape)
print(labels_clean.shape)


# # Neural net

# In[8]:


def run_NN_l2(l2norm=0.01):

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
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
    print("L2 norm, accuracy: ", l2norm, accuracy)
    
    return accuracy


# In[9]:


[run_NN_l2(float(10)**x) for x in np.arange(-7, 1, 1)]


# In[10]:


def run_NN_l1(l1norm=0.01):

    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=0.3, random_state = 42)

    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(l1norm), input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(l1norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(l1norm)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(l1norm)))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
    print("L1 norm, accuracy: ", l1norm, accuracy)
    
    return accuracy


# In[11]:


[run_NN_l1(float(10)**x) for x in np.arange(-7, 1, 1)]


# In[12]:


def run_NN_l1_l2(penalty=0.01):

    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=0.3, random_state = 42)

    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=penalty, l2=penalty), input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=penalty, l2=penalty)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=penalty, l2=penalty)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=penalty, l2=penalty)))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
    print("L1_L2 norm, accuracy: ", penalty, accuracy)

    return accuracy


# In[13]:


[run_NN_l1_l2(penalty=float(10)**x) for x in np.arange(-7, 1, 1)]


# In[14]:


def run_NN_reg(kernel_regularizer=keras.regularizers.l2(0.01)):

    X_train, X_test, y_train, y_test =         train_test_split(features, labels, test_size=0.3, random_state = 42)

    # Sequential model, 7 classes of output.
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer, input_dim=359))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dense(7, activation='softmax'))

    # Early stopping condition.
    callback = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0)]

    # Recompile model and fit.
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callback, verbose=False)

    # Check accuracy.
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]
    print("kernel_reg, constant, accuracy: ", kernel_regularizer, reg_constant, accuracy)
    
    return accuracy


# In[15]:


run_NN_reg(keras.regularizers.l2(0.01))


# In[ ]:


run_NN_reg(keras.regularizers.l1(0.01))

