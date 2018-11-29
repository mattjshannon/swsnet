
# coding: utf-8

# # Make predictions of stellar 'group' on CASSIS spectra

# Trained on SWS Atlas data.

# In[1]:


import glob

import pandas as pd
import tensorflow as tf

from tensorflow import keras
from swsnet.dataframe_utils import read_spectrum


# In[2]:


def load_model(file_path):
    """Returns a keras model (compressed as .h5)."""
    try:
        model = keras.models.load_model(file_path)
    except Exception as e:
        raise e
        
    return model


# ## Load keras model

# Stored as .h5 file.

# In[3]:


model = load_model('sws_model_01.h5')
model.summary()


# ## Read in metadata (pd.DataFrame)

# In[4]:


data_dir = '../../data/cassis/'
metadata_pickle = data_dir + 'metadata_step1_normalized.pkl'
meta = pd.read_pickle(metadata_pickle)


# In[5]:


meta.head()


# # Perform predictions

# In[40]:


def predict_group(spectrum):
    """Return the probabilities (from model) that source belongs to each group."""
    f = spectrum['flux'].values
    probabilities = model.predict(np.array([f]))
    return probabilities


# In[70]:


results_list = []

# Iterate over all spectra.
for index, row in enumerate(meta.itertuples()):
    if index % 200 == 0:
        print(index)
    
    file_path = getattr(row, 'file_path')
    aorkey = getattr(row, 'aorkey')

    spectrum = read_spectrum(data_dir + file_path)
    probabilities = predict_group(spectrum)
    
    wrap = [index, aorkey, file_path, *list(*probabilities)]
    results_list.append(wrap)
    
print('Done.')


# In[71]:


results_list[0]


# In[72]:


np.savetxt('results.txt', np.array(results_list), delimiter=',', fmt='%s',
           header='index, aorkey, file_path, PROBABILITIES (groups 0 - 4) shifted by one downwards.')

