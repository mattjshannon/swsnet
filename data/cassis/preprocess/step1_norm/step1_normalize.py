
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mattpy.utils import smooth
from swsnet.norm_utils import normalize_spectrum, renormalize_spectrum


# # Read metadata in

# In[14]:


meta = pd.read_pickle('../metadata.pkl')


# In[15]:


nrows = meta.shape[0]
nrows


# In[16]:


meta


# # Determine normalization parameters (and plot-opt)

# In[19]:


determine_parameters = True

def norm_and_plot(meta):
    param_list = []
    
    for index, filename in enumerate(meta['file_path']):
        if index % 200 == 0:
            print(index, ' / ', nrows)

        # Full classifier
        try:
            classifier = meta['full_classifier'][index]
        except Exception as e:
            classifier = ''
            
        # Perform shift/renormalization
        parameters = normalize_spectrum(filename, classifier,
                                        plot=False, verbose=False)
        
        # Save parameters to a list
        spec_min, spec_max, norm_factor = parameters
        param_list.append([filename, *parameters])
        
    return param_list


# In[20]:


if determine_parameters:
    par_list = norm_and_plot(meta)
    header = 'iso_filename, spec_min, spec_max, norm_factor (shift first, then norm!!)'
    np.savetxt('step1_norm_params.txt', par_list, delimiter=',', fmt='%s',
               header=header)


# ### Confirm we can read them back in later.

# In[21]:


norm_params = np.loadtxt('../step1_norm/step1_norm_params.txt', delimiter=',', dtype='str')
nrows = norm_params.shape[0]


# In[22]:


spectra_paths = norm_params.T[0]
spectra_paths


# In[23]:


norm_params[0]


# # Perform normalization

# In[25]:


file_path_list = []

for index, file_path in enumerate(spectra_paths):
    if index % 100 == 0:
        print(index, ' / ', nrows)
    
    # Normalization parameters for this spectrum:
    norm_factors = norm_params[index]
    
    # Renormalize, save to new pickle.
    save_path = renormalize_spectrum(file_path, norm_factors, verbose=False)
    
    # Do something with meta dataframe??
    file_path_list.append(save_path)
    
#     if index >= 10:
#         break


# In[26]:


def update_dataframe(meta, file_path_list):
    
    def check_tdts(old_file_paths, new_file_paths):

        old_list = [x.split('/')[-1].split('.pkl')[0] for x in old_file_paths]
        new_list = [x.split('/')[-1].split('_')[0] for x in new_file_paths]
        
        if old_list != new_list:
            raise SystemExit("TDTs don't match.")

        return    
    
    # Make a copy of the dataframe.
    new_meta = meta.copy()
    
    # Isolate file_path from meta dataframe.
    old_file_paths = meta['file_path']
    new_file_paths = file_path_list
    
    # Compare them by TDT as a sanity check.
    check_tdts(old_file_paths, new_file_paths)
    
    # Update paths.
    new_meta['file_path'] = new_file_paths
    
    # Save to disk.
    new_meta.to_pickle('../metadata_step1_normalized.pkl')
    print('Saved: ', '../metadata_step1_normalized.pkl')
    
    return new_meta


# In[27]:


new_meta = update_dataframe(meta, file_path_list)


# In[28]:


meta.head()


# In[29]:


new_meta.head()


# ## Reindex dataframe, save again

# In[30]:


# SORT BY TDT!
df = new_meta

df['aorkey'] = df['aorkey'].astype(int)
df = df.sort_values(by=['aorkey'], ascending=True)
df = df.reset_index(drop=True)


# In[31]:


df.head()


# In[32]:


# Remove rows of objects not pickled (typically due to a data error).
bool_list = []
for path in df['file_path']:
    if os.path.isfile('../../' + path):
        bool_list.append(True)
    else:
        bool_list.append(False)

df = df.assign(data_ok=bool_list)
df = df.query('data_ok == True')


# In[33]:


df.head()


# In[34]:


df.to_pickle('../metadata_step1_normalized.pkl')
print('Saved: ', '../metadata_step1_normalized.pkl')

