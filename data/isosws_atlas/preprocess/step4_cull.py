
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy
import pandas as pd


# In[8]:


meta = pd.read_pickle('metadata_step1_sorted.pkl')


# In[9]:


meta


# In[ ]:


# Make a list of the entries for which data_ok should == False.
# Based on a multipage PDF of all the spectra (starting at page=1).
# Confirm the TDTs match afterward.
pdf_pages_for_bad_data = [
    9, 11, 14, 15, 22, 23,
    26, 29, 30, 33, 46, 50,
    51, 58, 59, 62, 66, 67,
    71, 83, 84, 88, 89, 90,
    93, 94, 95, 122, 124, 132,
    136, 148, 152, 155, 160, 161,
    162, 163, 165, 170, 184, 191,
    194, 197, 206, 209, 211, 253
    
]

# maybes...
# 19, 32, 34, 54, 74,
# 76, 80, 92, 105, 131,
# 190, 195, 200, 203, 213,
# 229, 

