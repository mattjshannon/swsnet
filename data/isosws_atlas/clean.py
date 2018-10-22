
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy
import pandas as pd


# In[4]:


meta = pd.read_pickle('metadata.pkl')


# In[5]:


meta.describe()


# In[6]:


meta.head()


# In[29]:


for index, filename in enumerate(meta['file_path']):
    if index % 20 == 0:
        print(index)
    spec = pd.read_pickle(filename)
    plt.plot(spec.wavelength, spec.flux)
    plt.title(filename + ', data ok=' + str(meta['data_ok'][index]))
    plt.axhline(y=0, color='red', ls='--')
    plt.savefig('plots/' + filename.split('/')[-1] + '.pdf')    
    plt.close()
#     if index >= 10:
#         break


# In[21]:




