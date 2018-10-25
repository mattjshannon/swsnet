
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mattpy.utils import smooth


# In[10]:


def save_shift_figure(plot_dir, classifier, iso_filename, wave, flux,
                      smooth_flux, smooth_flux_shift, verbose=True):
    """Saves a .PDF figure of the raw spectrum and shifted smoothed spectrum."""
    
    # Plot the raw spectrum and smoothed spectrum.
    plt.plot(wave, flux)
    plt.plot(wave, smooth_flux)
    plt.axhline(y=0, color='k', ls='--', lw=1)
    plt.axhline(y=np.nanmax(smooth_flux_shift), ls='--', lw=1,
                zorder=-10, color='k')
    
    # Save to disk.
    plt.title(iso_filename + ' - classifier: ' + classifier)
    savepath = plot_dir + iso_filename + '_A.pdf'
    plt.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.close()
    
    if verbose:
        print('Saved: ', savepath)
    
    return


def save_renorm_figure(plot_dir, classifier, iso_filename, renorm_wave,
                       renorm_flux_shift, smooth_flux_shift,
                       verbose=True):
    """Saves a .PDF figure of the renormalized spectrum."""
    
    # Plot the renormalized/shifted spectrum.
    plt.plot(renorm_wave, renorm_flux_shift)
    plt.plot(renorm_wave, smooth_flux_shift)
    plt.axhline(y=0, color='red', ls='--', lw=1)
    plt.axhline(y=1, ls='--', lw=1, zorder=-10, color='red')
    
    # Save to disk.
    plt.title(iso_filename + ' - classifier: ' + classifier)
    savepath = plot_dir + iso_filename + '_B.pdf'
    plt.savefig(savepath, format='pdf', bbox_inches='tight')
    plt.close()
    
    if verbose:
        print('Saved: ', savepath)
    
    return


# In[11]:


def read_spectrum(file_path):
    """Returns an ISO spectrum (wave, flux, etc.) from a pickle."""
    spectrum = pd.read_pickle(file_path)
    
    wave = spectrum['wavelength']
    flux = spectrum['flux']
    specerr = spectrum['spec_error']
    normerr = spectrum['norm_error']
    fluxerr = specerr + normerr
    
    return wave, flux, fluxerr


def smooth_spectrum(flux, **kwargs):
    """Returns a shifted, smoothed spectrum, ready for normalization."""
    spec_min = np.nanmin(flux)
    spec_max = np.nanmax(flux)
    # print(spec_min, spec_max)

    # Smooth it to find the general low/high points.    
    smooth_flux = smooth(flux.values, **kwargs)
    
    # Smooth it to find the general low/high points, and shift to zero.
    smooth_flux_shift = smooth(flux.values - spec_min, **kwargs)

    # Upper normalization factor (s.t. the maximum of the continuum is 1.0).
    norm_factor = np.nanmax(smooth_flux_shift)

    return spec_min, spec_max, smooth_flux, smooth_flux_shift, norm_factor


def normalize_spectrum(file_path, classifier, plot=True, verbose=True):
    """Normalizes an ISO spectrum to span 0-1 (the main curvature)."""
    wave, flux, fluxerr = read_spectrum('../' + file_path)
    
    # Shift spectrum and get normalization factors.
    # Minimum should now=0.0.
    spec_min, spec_max, smooth_flux, smooth_flux_shift, norm_factor =         smooth_spectrum(flux, window_len=40)

    # Final renormalized quantities.
    renorm_wave = wave
    renorm_flux_shift = (flux - spec_min) / norm_factor
    
    # Plotting directory
    plot_dir = 'step2_norm/plots/'
    
    # Save file name.
    iso_filename = file_path.split('/')[-1].split('.pkl')[0]
    
    if plot:
        # Save a figure showing the initial smooth/shift.
        save_shift_figure(plot_dir, classifier, iso_filename,
                          wave, flux, smooth_flux, smooth_flux_shift,
                          verbose=False)

        # Save a figure of the final renormalized spectrum.
        save_renorm_figure(plot_dir, classifier, iso_filename,
                           renorm_wave, renorm_flux_shift,
                           smooth_flux_shift/norm_factor,
                           verbose=False)
    
    return spec_min, spec_max, norm_factor


# In[12]:


meta = pd.read_pickle('step1_sort/metadata_step1_sorted.pkl')


# In[13]:


nrows = meta.shape[0]
meta;


# In[18]:


def norm_and_plot(meta):
    param_list = []
    
    for index, filename in enumerate(meta['file_path']):
        if index % 200 == 0:
            print(index, ' / ', nrows)

        # Full classifier
        classifier = meta['full_classifier'][index]
            
        # Perform shift/renormalization
        parameters = normalize_spectrum(filename, classifier,
                                        plot=True, verbose=False)
        
        # Save parameters to a list
        spec_min, spec_max, norm_factor = parameters
        param_list.append([filename, *parameters])
        
    return param_list


# In[19]:


par_list = norm_and_plot(meta)


# In[20]:


np.savetxt('step2_norm/step2_norm_params.txt', par_list, delimiter=',', fmt='%s',
           header='iso_filename, spec_min, spec_max, norm_factor (shift first, then norm!!)')

