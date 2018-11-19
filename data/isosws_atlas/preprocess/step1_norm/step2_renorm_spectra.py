
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


meta = pd.read_pickle('../metadata_step0.pkl')


# In[3]:


norm_params = np.loadtxt('../step1_norm/step1_norm_params.txt', delimiter=',', dtype='str')
nrows = norm_params.shape[0]


# In[4]:


spectra_paths = norm_params.T[0]
spectra_paths


# In[5]:


norm_params[0]


# ***

# In[6]:


def read_spectrum(filename):
    """Returns an ISO spectrum (wave, flux, etc.) from a pickle."""
    spectrum = pd.read_pickle(filename)
    
    wave = spectrum['wavelength']
    flux = spectrum['flux']
#     specerr = spectrum['spec_error']
#     normerr = spectrum['norm_error']
#     fluxerr = specerr + normerr
    fluxerr = spectrum['uncertainty']
    
    return wave, flux, fluxerr, spectrum


# In[11]:


def renormalize_spectrum(file_path, norm_factors, output_dir='../spectra_normalized/',
                         verbose=True):

    # Sanity check that the parameters are for this particular file.
    if file_path != norm_factors[0]:
        raise SystemExit('File paths do not match!')
    
    # Read the original pickled spectrum.
    full_file_path = '../../' + file_path
    wave, flux, fluxerr, spectrum = read_spectrum(filename=full_file_path)
    
    # Identify the scaling factors.
    _, spec_min, spec_max, norm_fac = norm_factors
    spec_min = float(spec_min)
    spec_max = float(spec_max)
    norm_fac = float(norm_fac)
    
    # Scale its flux using the norm factors.
    renorm_flux = (flux - spec_min) / norm_fac
    
    # Create a new pickle with the scaled spectrum (otherwise the same structure).
    spectrum['flux'] = renorm_flux
    
    # Save new pickle.
    save_path = file_path.replace('.pkl', '_renorm.pkl')
    save_path = save_path.replace('spectra', 'spectra_normalized')
    full_save_path = '../../' + save_path
    spectrum.to_pickle(full_save_path)
    
    # Print 'Saved!' statement if verbose.
    if verbose:
        print('Saved: ', full_save_path)
    
    return save_path


# ***

# In[13]:


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


# In[17]:


def check_tdts(old_file_paths, new_file_paths):
    
    old_list = [x.split('/')[-1].split('_')[0] for x in old_file_paths]
    new_list = [x.split('/')[-1].split('_')[0] for x in new_file_paths]
    
    if old_list != new_list:
        raise SystemExit("TDTs don't match.")
    
    return


# In[18]:


def update_dataframe(meta, file_path_list):
    
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


# In[19]:


new_meta = update_dataframe(meta, file_path_list)


# ***

# In[20]:


meta.head()


# In[21]:


new_meta.head()

