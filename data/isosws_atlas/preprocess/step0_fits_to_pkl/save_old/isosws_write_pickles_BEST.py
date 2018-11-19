
# coding: utf-8

# # ISO-SWS data preprocessing: convert to pickled dataframes

# In[1]:


import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from IPython.core.debugger import set_trace as st
from scipy.interpolate import splev, splrep
from spectres import spectres


# In[2]:


def downsample_to_cassis_interpolate(df):
    """Downsample to match the wavelength grid of CASSIS."""
    
    wave = df['wavelength']
    flux = df['flux']
    spec_error = df['spec_error']
    norm_error = df['norm_error']

    def spline(x, y, new_x):
        spline_model = splrep(x=x, y=y)
        new_y = splev(x=new_x, tck=spline_model)
        return new_y

    new_wave = cassis_wave
    new_flux = spline(wave, flux, new_wave)
    new_spec_error = spline(wave, spec_error, new_wave)
    new_norm_error = spline(wave, norm_error, new_wave)

    col_stack = np.column_stack([new_wave, new_flux, new_spec_error, new_norm_error])
    col_names = ['wavelength', 'flux', 'spec_error', 'norm_error']

    df2 = pd.DataFrame(col_stack, columns=col_names)
    
    return df2

def downsample_to_cassis_spectres(df):
    """Downsample to match the wavelength grid of CASSIS."""

    def spline(x, y, new_x):
        spline_model = splrep(x=x, y=y)
        new_y = splev(x=new_x, tck=spline_model)
        return new_y
      
    wave = df['wavelength']
    flux = df['flux']
    spec_error = df['spec_error']
    norm_error = df['norm_error']

    wave = wave.values
    flux = flux.values
    spec_error = spec_error.values
    norm_error = norm_error.values
    
    new_wave = cassis_wave
    new_flux, new_spec_error = spectres(new_spec_wavs=new_wave, old_spec_wavs=wave,
                                        spec_fluxes=flux, spec_errs=spec_error)
    _, new_norm_error = spectres(new_spec_wavs=new_wave, old_spec_wavs=wave,
                                 spec_fluxes=flux, spec_errs=norm_error)

    col_stack = np.column_stack([new_wave, new_flux, new_spec_error, new_norm_error])
    col_names = ['wavelength', 'flux', 'spec_error', 'norm_error']

    df2 = pd.DataFrame(col_stack, columns=col_names)
    
    return df2


# In[3]:


# Some useful functions....
cassis_wave = np.loadtxt('isosws_misc/cassis_wavelength_grid.txt', delimiter=',')

def convert_fits_to_pickle(path, verify_pickle=False, verbose=False, match_cassis_wavegrid=False):
    """Full conversion from ISO-SWS <filename.fits to <filename>.pkl, which contains a pd.DataFrame.
    
    Args:
        path (str): Path to <filename>.fits file (of an ISO-SWS observation).
        verify_pickle (bool): Confirm the pickle was succesful created; does so by comparing the
            pd.DataFrame before and after writing the pickle.
        
    Returns:
        True if successful.
        
    Note:
        DataFrame can be retrieved from the pickle by, e.g., df = pd.read_pickle(pickle_path).
    """
    
    if verbose:
        print('Pickling: ', path)
    
    # Convert .fits file to pandas DataFrame, header.Header object.
    try:
        df, header = isosws_fits_to_dataframe(path)
    except Exception as e:
        raise(e)

    # Downsample to match the CASSIS wavegrid if desired.
    if match_cassis_wavegrid:
#         df = downsample_to_cassis_interpolate(df)
        df = downsample_to_cassis_spectres(df)
     
    # Determine the pickle_path to save to. Being explicit here to 'pickle_path' is clear.
    base_filename = path.replace('.fit', '.pkl').split('/')[-1]
    
    # Save the dataframe to a pickle.
    pickle_path = 'spectra/' + base_filename
    df.to_pickle(pickle_path)
    
    if verbose:
        print('...saved: ', pickle_path)

    # Test dataframes for equality before/after pickling if verify_pickle == True.
    if verify_pickle:
        tmp_df = pd.read_pickle(pickle_path)
        if df.equals(tmp_df):
            if verbose:
                print()
                print('DataFrame integrity verified -- pickling went OK!')
                print()
        else:
            raise ValueError('Dataframes not equal before/after pickling!')
    
    return pickle_path


def isosws_fits_to_dataframe(path, test_for_monotonicity=True):
    """Take an ISO-SWS .fits file, return a pandas DataFrame containing the data (with labels) and astropy header.
    
    Args:
        path (str): Path of the .fits file (assumed to be an ISO-SWS observation file).
        test_for_monotonicity (bool, optional): Check that the wavelength grid is monotinically increasing.
        
    Returns:
        df (pd.DataFrame): Pandas dataframe with appropriate labels (wavelength, flux, etc.).
        header (astropy.io.fits.header.Header): Information about observation from telescope.
        
    Note:
        Header can be manipulated with, e.g., header.totextfile(some_path).
        See http://docs.astropy.org/en/stable/io/fits/api/headers.html.
    """
    
    def monotonically_increasing(array):
        """Test if a list has monotonically increasing elements. Thank you stack overflow."""
        return all(x < y for x, y in zip(array, array[1:]))
    
    # Read in .fits file.
    hdu = fits.open(path)
    
    # Retrieve the header object.
    header = hdu[0].header
    
    # Extract column labels/descriptions from header.
    # Can't do this because the header is not well-defined. That's OK, hard-coded the new column names below.
    
    # Convert data to pandas DataFrame.
    dtable = Table(hdu[0].data)
    df = dtable.to_pandas()
    
    # Convert the nondescriptive column labels (e.g., 'col01def', 'col02def') to descriptive labels.
    old_keys = list(df.keys())
    new_keys = ['wavelength', 'flux', 'spec_error', 'norm_error']
    mydict = dict(zip(old_keys, new_keys))
    df = df.rename(columns=mydict)  # Renamed DataFrame columns here.
    
    if test_for_monotonicity:
         if not monotonically_increasing(df['wavelength']):
                raise ValueError('Wavelength array not monotonically increasing!', path)
    
    return df, header


# ***

# ## Find out how many files we're working with

# In[7]:


spec_dir = 'fits_spectra/'
spec_files = np.sort(glob.glob(spec_dir + '*.fit'))


# In[8]:


len(spec_files)


# ## Build dataframe containing metadata (including labels) and paths to pickled files.

# ###### Creates isosws_metadata_df.pkl.

# In[68]:


# Only do this once.
recreate_meta_pickle = True

if recreate_meta_pickle:
    def create_swsmeta_dataframe():
        """Create a dataframe that contains the metadata for the ISO-SWS Atlas."""
        
        def simbad_results():
            """Create a dictionary of the SIMBAD object type query results."""
            simbad_results = np.loadtxt('isosws_misc/simbad_type.csv', delimiter=';', dtype=str)
            simbad_dict = dict(simbad_results)
            return simbad_dict
        
        def sexagesimal_to_degree(tupe):
            """Convert from hour:minute:second to degrees."""
            sex_str = tupe[0] + ' ' + tupe[1]
            c = SkyCoord(sex_str, unit=(u.hourangle, u.deg))
            return c.ra.deg, c.dec.deg        
        
        def transform_ra_dec_into_degrees(df):
            """Perform full ra, dec conversion to degrees."""
            ra = []
            dec = []
            for index, value in enumerate(zip(df['ra'], df['dec'])):
                ra_deg, dec_deg = sexagesimal_to_degree(value)
                ra.append(ra_deg)
                dec.append(dec_deg)
            df = df.assign(ra=ra)
            df = df.assign(dec=dec)
            return df

        # Read in the metadata
#         meta_filename = 'isosws_misc/kraemer_class.csv'
        meta_filename = 'isosws_misc/kraemer_class_fixed.csv'
        swsmeta = np.loadtxt(meta_filename, delimiter=';', dtype=str)
        df = pd.DataFrame(swsmeta[1:], columns=swsmeta[0])
        
        # Add a column for the pickle paths (dataframes with wave, flux, etc).
        pickle_paths = ['spectra/' + x.zfill(8) + '_sws.pkl' for x in df['tdt']]
        df = df.assign(file_path=pickle_paths)
        
        # Add a column for SIMBAD type, need to query 'simbad_type.csv' for this. Not in order naturally...
        object_names = df['object_name']
        object_type_dict = simbad_results()
        object_types = [object_type_dict.get(key, "empty") for key in object_names]
        df = df.assign(object_type=object_types)

        # Transform ra and dec into degrees.
        df = transform_ra_dec_into_degrees(df)
        
        # Remove rows of objects not pickled (typically due to a data error).
        bool_list = []
        for path in df['file_path']:
            if os.path.isfile(path):
                bool_list.append(True)
            else:
                bool_list.append(False)
                
        df = df.assign(data_ok=bool_list)
        df2 = df.query('data_ok == True')
        
        return df2
    
    df = create_swsmeta_dataframe()
    df.reset_index(drop=True, inplace=True)
    df.to_pickle('metadata.pkl')


# In[69]:


df.head()


# In[70]:


df


# In[71]:


np.unique(df['group'].values)


# In[67]:


df.describe()


# In[19]:


mdf = pd.read_pickle('isosws_metadata_df.pkl')
# mdf


# ## Convert spectra to dataframes and save to disk as pickles

# Requirement: must be in df['file_path']

# In[72]:


perform_conversion = True


# In[73]:


def is_in_meta_pickle(df, fits_file):

    tdt = fits_file.split('/')[-1].split('_sws.fit')[0]
    tdt_zfill = [x.zfill(8) for x in df['tdt'].values]
    
    if tdt in tdt_zfill:
        return True
    else:
        return False


# In[74]:


# Note the break I've added; remove for full conversion.
if perform_conversion:
    
    n_skipped = 0
    print('=============================\nConverting fits files...\n=============================\n')

    # Iterate over all the fits files and convert them.
    for index, fits_file in enumerate(spec_files):
        
        if not is_in_meta_pickle(df, fits_file):
            print(fits_file, 'Skipping.')
            n_skipped += 1
            continue
        
#         if index >= 22:
#             break

        if index % 50 == 0:
            print(index, '/', len(spec_files))

        try:
            pickle_path = convert_fits_to_pickle(fits_file, verify_pickle=True,
                                                 verbose=False, match_cassis_wavegrid=True)
        except Exception as e:
            print(e)
            print(fits_file, 'EXCEPTION!')
            n_skipped += 1
            continue

    print('\n=============================\nComplete.\n=============================')
    
    print('Number of spectra skipped due to missing monotonicity: ', n_skipped)


# ***

# ***

# ***

# # Appendix A -- Example transformation from .fits to pd.dataframe

# #### Convert spectrum file to dataframe, header

# In[8]:


# Grab the first file from the glob list.
test_spec = spec_files[0]
test_spec


# In[9]:


# Read it in with astropy.io.fits, check dimensions.
test_hdu = fits.open(test_spec)
test_hdu.info()


# In[68]:


# Utilize our defined function to transform a string of the .fits filename to a pandas dataframe and header.
# 'header' will be an astropy.io.fits.header.Header object; see a couple subsections below for conversion options.
df, header = isosws_fits_to_dataframe(test_spec)


# #### Inspect dataframe

# In[69]:


df.shape


# In[70]:


df.head()


# In[71]:


df.describe()


# #### Header from the .fits file

# In[72]:


type(header)


# In[73]:


# Uncomment below to see full header of one file as an example.
header


# In[74]:


# Can convert to other formats if we want to use the header information for something.
# See http://docs.astropy.org/en/stable/io/fits/api/headers.html

# header_str = header.tostring()
# header.totextfile('test_header.csv')


# # Compare smoothing windows

# In[37]:


wave = test_hdu[0].data.T[0]
flux = test_hdu[0].data.T[1]
fluxerr = test_hdu[0].data.T[2]

cassis_wave = np.loadtxt('isosws_misc/cassis_wavelength_grid.txt', delimiter=',')


# In[39]:


from spectres import spectres
# spectres.spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None)
downsamp_wave = cassis_wave
downsamp_flux, downsamp_fluxerr = spectres(new_spec_wavs=cassis_wave, old_spec_wavs=wave, spec_fluxes=flux, spec_errs=fluxerr)


# In[40]:


plt.plot(wave, flux, label='raw');
plt.plot(downsamp_wave, downsamp_flux, label='spectres downsamp')
# plt.plot(wave, smooth(flux, window_len=100), label='hanning');
# plt.plot(wave, smooth(flux, window_len=100), label='flat');
# plt.plot(wave, smooth(flux, window_len=100), label='hamming');
# plt.plot(wave, smooth(flux, window_len=100), label='bartlett');
# plt.plot(wave, smooth(flux, window_len=100), label='blackman');

# plt.xlim(xmin=10.45, xmax=10.65)
# plt.ylim(ymax=1500)|
plt.legend(loc=0)


# In[44]:


np.all(downsamp_wave == cassis_wave)

