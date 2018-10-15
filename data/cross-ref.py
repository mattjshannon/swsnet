
# coding: utf-8

# ## Identify matches between the SWS/CASSIS datasets.

# In[1]:


from concurrent.futures import ProcessPoolExecutor

import astropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import swsnet


# In[2]:


def get_ang_sep(ra1, dec1, ra2, dec2):
    """Calculate the angular distance between two coordinates."""
    
    delt_lon = (ra1 - ra2)*np.pi/180.
    delt_lat = (dec1 - dec2)*np.pi/180.
    # Haversine formula
    dist = 2.0*np.arcsin( np.sqrt( np.sin(delt_lat/2.0)**2 +          np.cos(dec1*np.pi/180.)*np.cos(dec2*np.pi/180.)*np.sin(delt_lon/2.0)**2 ) )  

    ang_sep = dist/np.pi*180.
     
    return ang_sep


# In[3]:


def get_matches(in_tuple):
    """Find sources common to CASSIS/SWS Atlas."""
    
    global czip
    global c_aor
    global s_tdt
    
    sindex, sra, sdec = in_tuple
    match_list = []
    
    for index, value in enumerate(czip):
        cindex, cra, cdec = value
        ang_sep = get_ang_sep(sra, sdec, cra, cdec)
        if ang_sep*3600 <= 11:
            positive_match = c_aor[cindex], s_tdt[sindex], ang_sep*3600, cindex, sindex
            match_list.append(positive_match)
    
    if match_list:
        return match_list
    else:
        return None


# In[4]:


cass = pd.read_pickle('cassis/metadata.pkl')
sws = pd.read_pickle('isosws_atlas/metadata.pkl')

cra = cass.ra.values
cdec = cass.dec.values

sra = sws.ra.values
sdec = sws.dec.values

c_aor = cass.aorkey.values
s_tdt = sws.tdt.values


# Array containing all possible combinations:

# In[5]:


czip = []
szip = []

global czip
global c_aor
global s_tdt

for cindex, cvalue in enumerate(cra):
    czip.append((cindex, cra[cindex], cdec[cindex]))
    
for sindex, svalue in enumerate(sra):
    szip.append((sindex, sra[sindex], sdec[sindex]))    


# In[6]:


len(czip), len(szip)


# ### Use parallelization to calculate.

# In[7]:


pool = ProcessPoolExecutor(max_workers=14)
results = list(pool.map(get_matches, szip))


# In[8]:


def clean(array):
    """Tidy up the results from get_matches."""
    
    cleaned_array = []
    array_remove_nones = [x for x in array if x is not None]
    
    for entry in array_remove_nones:
        if len(entry) == 1:
            cleaned_array.append(entry[0])
        else:
            for subentry in entry:
                cleaned_array.append(subentry)
                
    return np.array(cleaned_array)


# In[9]:


cleaned = clean(results)


# In[10]:


# aor, tdt, ang_sep (''), cassis_index, sws_index
cleaned


# In[11]:


cleaned.shape


# In[12]:


ang_seps = cleaned.T[2].astype(float)


# In[13]:


plt.hist(ang_seps);
plt.xlabel('angular separation (arsec)')
plt.ylabel('counts')


# In[20]:


np.savetxt('matches/matches_within_11arcsec.txt', cleaned, delimiter=',', fmt='%s',
           header='AOR, TDT, ang_sep (''), cassis_index, sws_index')

