
# coding: utf-8

# # Manual cull of untrustworthy sources, classes?

# Applied after step 4, which removes group 7 and any with the : or :: flags, i.e. uncertain classification.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


meta = pd.read_pickle('step4_cull_uncertain/metadata_sorted_normalized_culled.pkl')


# In[3]:


meta

