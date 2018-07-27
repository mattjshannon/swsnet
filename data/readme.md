I've included individual `README.txt` files in the `cassis` and `isosws_atlas` folders.

But generally, each folder contains the following:

- astronomical spectra in the form of pandas dataframes (stored as pickles). These are in zip files for ease of transportation.
- metadata about the above spectra, including the paths to the corresponding data file (see *_metadata_df.pkl).
- a couple extraneous things, including some jupyter notebooks and miscellaneous files. You can safely ignore these if you'd like, as the above are all that really matters.

I have downsampled the ISO data from very high resolution (almost 50000 wavelength points per spectrum) to that of Spitzer/CASSIS (~360 I believe). All data should be on the same wavelength grid at this point if I haven't made a mistake.

Matt
