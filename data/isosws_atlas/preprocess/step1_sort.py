
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


# In[3]:


meta = pd.read_pickle('../metadata.pkl')


# In[4]:


meta.describe()


# In[5]:


meta.head()


# In[6]:


meta['tdt'] = meta['tdt'].astype(int)


# In[9]:


meta = meta.sort_values(by=['tdt'], ascending=True)


# In[10]:


meta


# In[14]:


meta = meta.reset_index(drop=True)


# In[15]:


meta


# In[16]:


meta.to_pickle('metadata_step1_sorted.pkl')

