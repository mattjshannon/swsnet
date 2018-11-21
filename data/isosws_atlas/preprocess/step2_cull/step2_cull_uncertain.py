
# coding: utf-8

# # Remove untrustworthy sources

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


meta = pd.read_pickle('../metadata_step1_normalized.pkl')


# In[3]:


meta


# 
# First let's cull based on the uncertainty flag, and if it's group 7. i.e., for those data, let's set the ``data_ok`` flag to ``False``.

# ### Manual quality flags

# In[ ]:


def apply_manual_flags(df):
    return df


# In[12]:


tdts, flags = np.loadtxt('quality_flags.txt', delimiter=',').T.astype(int)


# In[15]:


flags


# In[24]:


plt.hist(flags, width=0.5)
plt.xlim(right=2.5)


# ### Exclude group 7 and uncertain data

# In[34]:


updated_data_ok = []

for row in meta.itertuples():
    index = row[0]

    # Attributes needed for determining quality of spectrum.
    tdt = getattr(row, "tdt")
    group_num = getattr(row, "group")
    flag = getattr(row, "uncertainty_flag")
    
    # Don't trust group 7, or anything with a flag.
    if (int(group_num) == 7) or (flag != ''):
        trust_tdt = False
    else:
        trust_tdt = True
        
    # Don't trust manually flagged spectra.
    tdt_index = np.where(tdts == tdt)[0][0]
    manual_flag = flags[tdt_index]
    if manual_flag == 0:
        trust_tdt = True
    else:
        trust_tdt = False        

    updated_data_ok.append(trust_tdt)
        
print('Number still OK: ', sum(updated_data_ok))


# ### Save to disk as a new pickle file

# In[35]:


meta2 = meta.copy()


# In[36]:


meta2.head()


# In[37]:


meta2['data_ok'] = updated_data_ok


# In[38]:


meta2


# In[39]:


meta2.to_pickle('../metadata_step2_culled.pkl')


# In[41]:


np.sum(meta2['data_ok'].values)

