
# coding: utf-8

# # ISO-SWS data preprocessing: check wavelength grids

# In[8]:


import glob
import numpy as np
import pandas as pd


# In[9]:


# Some useful functions....

def get_wavelength_grid(path):
    """Return the wavelength grid (as a numpy array) from a pickled pd.DataFrame file."""
    df = pd.read_pickle(path)
    
#     print(df.keys())
#     return True
    wavelength_array = df['wavelength']
    return wavelength_array


# ***

# ## Wavelength arrays

# In[10]:


pickle_dir = 'isosws_dataframes/'
pickle_files = np.sort(glob.glob(pickle_dir + '*.pkl'))


# In[11]:


len(pickle_files)


# In[12]:


for index, filename in enumerate(pickle_files):
    wave = get_wavelength_grid(filename)
    if index == 0:
        static_wave = wave
        continue
#     if index >= 100:
#         break
        
    if index % 200 == 0:
        print(index, '/', len(pickle_files))
        
    if not np.all(static_wave == wave):
        raise ValueError('Wavelength arrays not equal...!')
    


# ***

# ***

# ***
