
# coding: utf-8

# # ISO-SWS data preprocessing: convert to pickled dataframes

# In[1]:


import glob
import numpy as np
import os
import pandas as pd

from astropy.io import fits
from astropy.table import Table

from swsnet.dataframe_utils import     isosws_fits_to_dataframe, convert_fits_to_pickle, create_swsmeta_dataframe


# ***

# ## Find out how many files we're working with

# In[2]:


spec_dir = 'sws_irs/'
spec_files = np.sort(glob.glob(spec_dir + '*.fit'))


# In[3]:


len(spec_files)


# ### Identify the 'good' TDTs

# In[4]:


good_tdts = np.loadtxt('good_tdts.txt').astype(int)


# In[5]:


good_tdts


# In[6]:


len(good_tdts)


# ## Convert spectra to dataframes and save to disk as pickles

# In[13]:


perform_conversion = True


# In[14]:


# Note the break I've added; remove for full conversion.
if perform_conversion:
    print('=============================\nConverting fits files...\n=============================\n')

    # Iterate over all the fits files and convert them.
    for index, fits_file in enumerate(spec_files):
    
        tdt = int(fits_file.split('/')[-1].split('_irs')[0])
        if tdt not in good_tdts:
            continue
        
        if index % 20 == 0:
            print(index, '/', len(spec_files))

        pickle_path = convert_fits_to_pickle(fits_file, verify_pickle=True, verbose=False)

    print('\n=============================\nComplete.\n=============================')


# ## Build dataframe containing metadata (including labels) and paths to pickled files.

# ###### Creates isosws_metadata_df.pkl.

# In[7]:


# Only do this once.
recreate_meta_pickle = True


# In[8]:


if recreate_meta_pickle:
    df = create_swsmeta_dataframe(good_tdts=good_tdts)


# In[9]:


df


# In[10]:


np.unique(df['group'].values)


# In[11]:


np.unique(df['subgroup'].values)


# In[12]:


np.unique(df['uncertainty_flag'].values)


# In[13]:


np.unique(df['data_ok'].values)


# In[14]:


np.unique(df['object_type'].values)


# In[15]:


df.to_pickle('../metadata_step0.pkl')


# In[16]:


df.head()


# In[17]:


mdf = pd.read_pickle('../metadata_step0.pkl')
mdf


# ***

# ***

# ***

# # Appendix A -- Example transformation from .fits to pd.dataframe

# #### Convert spectrum file to dataframe, header

# In[63]:


# Grab the first file from the glob list.
test_spec = spec_files[0]
test_spec


# In[64]:


# Read it in with astropy.io.fits, check dimensions.
test_hdu = fits.open(test_spec)
test_hdu.info()


# In[68]:


# Utilize our defined function to transform a string of the .fits filename to a pandas dataframe and header.
# 'header' will be an astropy.io.fits.header.Header object; see a couple subsections below for conversion options.
df, header = isosws_fits_to_dataframe(test_spec)


# #### Inspect dataframe

# In[69]:


df.shape


# In[70]:


df.head()


# In[71]:


df.describe()


# #### Header from the .fits file

# In[72]:


type(header)


# In[73]:


# Uncomment below to see full header of one file as an example.
header


# In[74]:


# Can convert to other formats if we want to use the header information for something.
# See http://docs.astropy.org/en/stable/io/fits/api/headers.html

# header_str = header.tostring()
# header.totextfile('test_header.csv')


# ***
