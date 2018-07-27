This folder contains:
- astronomical spectra from the CASSIS project. These are pickled (.pkl) pandas DataFrames (see note below).
- a pickled metadata dataframe (cassis_metadata_df.pkl) with some relevant metadata and a pointer to the relevant pickle containing the spectrum.
- a couple of jupyter notebooks, but you don't really need to look through them. Basically just need to load the metadata pickled dataframe and unzip the
  cassis_dataframes*.zip files.
- a 'cassis_misc' folder with a paper in it.

Important note:

I've split the "cassis_dataframes" folder into 2 zip files to avoid the 100MB limit on GitHub. The contents of the following two zip files...

cassis_dataframes.zip
cassis_dataframes_spillover_README.zip

...should both be extracted into a "cassis_dataframes" folder.

Thx!
Matt
