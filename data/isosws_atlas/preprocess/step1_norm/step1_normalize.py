
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mattpy.utils import smooth
from swsnet.norm_utils import normalize_spectrum, renormalize_spectrum


# # Read metadata in

# In[2]:


meta = pd.read_pickle('../metadata_step0.pkl')


# In[3]:


nrows = meta.shape[0]
meta


# # Determine normalization parameters (and plot-opt)

# In[4]:


determine_parameters = False

def norm_and_plot(meta):
    param_list = []
    
    for index, filename in enumerate(meta['file_path']):
        if index % 200 == 0:
            print(index, ' / ', nrows)

        # Full classifier
        classifier = meta['full_classifier'][index]
            
        # Perform shift/renormalization
        parameters = normalize_spectrum(filename, classifier,
                                        plot=True, verbose=False)
        
        # Save parameters to a list
        spec_min, spec_max, norm_factor = parameters
        param_list.append([filename, *parameters])
        
    return param_list


# In[5]:


if determine_parameters:
    par_list = norm_and_plot(meta)
    header = 'iso_filename, spec_min, spec_max, norm_factor (shift first, then norm!!)'
    np.savetxt('step1_norm_params.txt', par_list, delimiter=',', fmt='%s',
               header=header)


# ### Confirm we can read them back in later.

# In[6]:


norm_params = np.loadtxt('../step1_norm/step1_norm_params.txt', delimiter=',', dtype='str')
nrows = norm_params.shape[0]


# In[7]:


spectra_paths = norm_params.T[0]
spectra_paths


# In[8]:


norm_params[0]


# # Perform normalization

# In[10]:


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


# In[11]:


def update_dataframe(meta, file_path_list):
    
    def check_tdts(old_file_paths, new_file_paths):

        old_list = [x.split('/')[-1].split('_')[0] for x in old_file_paths]
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
    new_meta.to_pickle('../metadata_normalized.pkl')
    print('Saved: ', '../metadata_normalized.pkl')
    
    return new_meta


# In[12]:


new_meta = update_dataframe(meta, file_path_list)


# In[13]:


meta.head()


# In[14]:


new_meta.head()

