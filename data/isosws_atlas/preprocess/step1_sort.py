
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy
import pandas as pd


# In[5]:


def create_plots():
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


# In[2]:


meta = pd.read_pickle('metadata.pkl')


# In[3]:


meta.describe()


# In[4]:


meta.head()


# In[16]:


meta['tdt'] = meta['tdt'].astype(int)


# In[19]:


meta.sort_values(by=['tdt'], ascending=True)


# In[20]:


meta.to_pickle('metadata_sorted.pkl')

