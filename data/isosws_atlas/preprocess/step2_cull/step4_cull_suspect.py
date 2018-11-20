
# coding: utf-8

# # Preprocessing notebook 5: extra manual culling of untrustworthy sources

# This is applied after step 4, which removes group 7 and any with the : or :: flags, i.e. uncertain classification.

# ***

# Necessary modules...

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

from swsnet.dataframe_utils import     ensure_exists, read_spectrum, read_metadata,     plot_spectrum, plot_dataframe


# ***

# ## 1. Load metadata

# Note that we start by using the metadata pickle which has set group 7 sources to data_ok=False, as well as any with uncertainty flags.

# In[4]:


meta = pd.read_pickle('step4_cull_uncertain/metadata_sorted_normalized_culled.pkl')


# In[5]:


meta.head()


# ***

# ## 2. Save PDFs by groupings

# Condition is `data_ok == True`.

# ### Plot all

# Save all of them to all.pdf: `pdftk *_renorm.pdf cat output all.pdf`

# In[7]:


save_dir = 'step5_cull_suspect/plots/'
pdf_file = save_dir + 'all.pdf'

# Plot all into a single PDF file.
with PdfPages(pdf_file) as pdfpages:   
    plot_dataframe(meta, save_dir=save_dir,
                   pdfpages=pdfpages, verbose=False)
    
print('Saved: ', pdf_file)


# ### Plot by group

# In[8]:


groups = ['1', '2', '3', '4', '5', '6']

# Save separate PDFs for each group in separate folders.
for index, group in enumerate(groups):

    # Dataframe subset by 'group' condition.
    meta_subset = meta.loc[meta['group'] == group]
    
    # Plot and save to a separate directory.
    save_dir = 'step5_cull_suspect/plots/'
    pdf_file = save_dir + 'group_' + str(group) + '.pdf'
    
    # Save all PDFs to one file.
    with PdfPages(pdf_file) as pdfpages:    
        plot_dataframe(meta_subset, save_dir=save_dir,
                       pdfpages=pdfpages, verbose=False)
    
    print('Saved: ', pdf_file)


# ### Plot by subgroup

# In[9]:


groups = ['1', '2', '3', '4', '5', '6']

# Iterate over all the groups.
for index, group in enumerate(groups):

    # Isolate group, identify subgroups.
    meta_subset = meta.loc[meta['group'] == group]
    subgroups = np.unique(meta_subset['subgroup']).astype(str)
    print()
    print('Subgroups of group', group, ': ', subgroups)
    
    # Iterate over all the subgroups.
    for subindex, subgroup in enumerate(subgroups):

        # Subgroup directory label, avoid weird characters.
        if subgroup == '':
            subgroup_label = 'empty'
        else:
            subgroup_label = subgroup.replace('/','_slash_')
            
        # Output directory.
        save_dir = 'step5_cull_suspect/plots/group' + group + '/'
        pdf_file = save_dir + 'subgroup_' + subgroup_label + '.pdf'
        ensure_exists(save_dir)        
        
        # Extract a single subgroup's subset from the meta DataFrame.
        meta_subsubset = meta_subset.loc[meta_subset['subgroup'] == subgroup]        
        
        # Plot all spectra in the subgroup.
        with PdfPages(pdf_file) as pdfpages:
            plot_dataframe(meta_subsubset, save_dir=save_dir,
                           pdfpages=pdfpages, verbose=False)
        
        print('Saved: ', pdf_file)      


# In[ ]:




