This folder contains:
- astronomical spectra from the ISO-SWS Atlas project. These are pickled (.pkl) pandas DataFrames. They have been significantly downsampled to
  match the resolution of the CASSIS data (think 50000 data points to 360!). This means some very narrow atomic lines have been lost in the conversion.
- a pickled metadata dataframe (isosws_metadata_df.pkl) with some relevant metadata and a pointer to the relevant pickle containing the spectrum.
- a couple of jupyter notebooks, but you don't really need to look through them. Basically just need to load the metadata pickled dataframe and unzip the
  isosws_dataframes.zip file.
- a 'isosws_misc' folder with a couple papers and some other bits (only really relevant for the notebooks).

Matt
