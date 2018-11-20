
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


meta = pd.read_pickle('../metadata_normalized.pkl')


# In[3]:


meta


# 
# First let's cull based on the uncertainty flag, and if it's group 7. i.e., for those data, let's set the ``data_ok`` flag to ``False``.

# ### Exclude group 7 and uncertain data

# In[5]:


updated_data_ok = []

for row in meta.itertuples():
    index = row[0]
    group_num = getattr(row, "group")
    flag = getattr(row, "uncertainty_flag")
    print(group_num)
    
#     if (int(group_num) == 7) or (flag != ''):
# #         print('bad! bad! ', index, group_num, flag)
#         updated_data_ok.append(False)
#     else:
#         updated_data_ok.append(True)
        
# print('Number still OK: ', sum(updated_data_ok))


# ### Save to disk as a new pickle file

# In[33]:


meta2 = meta.copy()


# In[34]:


meta2.head()


# In[36]:


meta2['data_ok'] = updated_data_ok


# In[38]:


meta2


# In[39]:


meta2.to_pickle('step4_cull/metadata_sorted_normalized_culled.pkl')

