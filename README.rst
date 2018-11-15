===============================
swsnet
===============================

.. image:: https://img.shields.io/travis/mattjshannon/swsnet.svg
        :target: https://travis-ci.org/mattjshannon/swsnet

.. image:: https://codecov.io/gh/mattjshannon/swsnet/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/mattjshannon/swsnet

.. image:: https://img.shields.io/badge/docs-available-brightgreen.svg
        :target: https://mattjshannon.github.io/swsnet/

Applying neural networks to the Sloan SWS astronomical dataset.

* Free software: 3-clause BSD license

Premise
--------
* Predict labels for the larger CASSIS dataset using the SWS dataset for training/validation.

Models
------
Models are presented in Jupyter notebooks (see folder ipy_notebooks). Primary focus right now is a neural network. *Please examine most recent notebook first!*

        - Attempt 06: https://github.com/mattjshannon/swsnet/blob/master/ipy_notebooks/keras_v06.ipynb
        
                - Pure neural network model
                - Tested the effect of changing test size vs. training size
                - Tested the effect of including/excluding the ISO group 7 ("fatally flawed/flux free") spectra; see relevant issue (https://github.com/mattjshannon/swsnet/issues/6).

        - Attempt 05: https://github.com/mattjshannon/swsnet/blob/master/ipy_notebooks/keras_v05.ipynb
        
                - Neural network, k-nearest neighbours, logistic regression
                - Improved the downsampling of the SWS (high res) data to the Spitzer (low res) wavelength array via https://github.com/ACCarnall/SpectRes. These changes are incorporated into the data pickles contained within unzip_me.zip
                - Added a cleaning step that excludes the group=7 classified SWS data (flux-free or fatally flawed); may in time wish to skip this, as it may be useful for our neural network to be able to identify these data types. However, as present, this improves the accuracy of logistic regression for this task.

        - Attempt 04: https://github.com/mattjshannon/swsnet/blob/master/ipy_notebooks/keras_v04.ipynb
                
                - Important bug-fix on the TF nueral network (accidentally trainined on full dataset instead of just training set!)
                - Updated SGD for k-nearest neighbours and logistic regression attempts in scikit-learn.
        
        - Attempt 03: https://github.com/mattjshannon/swsnet/blob/master/ipy_notebooks/keras_v03.ipynb
        
                - Reduced overfitting by keras.callback.EarlyStopping (based on no accuracy improvement over 4 consecutive NN iterations)
                - Much more thorough SGD attempt. Improved but stil no match for a neural network here. Will likely drop in seubsequent notebooks.

        - Attempt 02: https://github.com/mattjshannon/swsnet/blob/master/ipy_notebooks/keras_v02.ipynb
                
                - Accuracy improvement by normalizing the spectra prior to training.

        - Attempt 01: https://github.com/mattjshannon/swsnet/blob/master/ipy_notebooks/keras_v01.ipynb        



Dataset
-------
- 1239 labeled spectra -- Sloan ISO/SWS Atlas
        - Data: http://adsabs.harvard.edu/abs/2003ApJS..147..379S.
        - Labels: http://adsabs.harvard.edu/abs/2002ApJS..140..389K
- 6732 unlabeled spectra -- CASSIS Spitzer/IRS spectra
        - Data: https://cassis.sirtf.com/
- Note
        - The ISO data have been downsampled to the wavelength grid of the CASSIS data (a significant reduction, from R~3000 to R~100) as a first attempt.

Labels
------
The SWS data have the following labels:

- Object type
        - From SIMBAD (not always reliable)
- Group
        1. Naked stars
        2. Stars with dust
        3. Warm, dusty objects
        4. Cool, dusty objects
        5. Very red objects
        6. Continuum-free objects but having emission lines
        7. Flux-free and/or fatally flawed spectra
- Subgroup
        .. image:: docs/images/subgroup.png
        - see further classification details from http://adsabs.harvard.edu/abs/2002ApJS..140..389K.
- Suffix
        .. image:: docs/images/subgroup_suffix.png
- Example
        - W Cet has classifier 2.SEa:
        - i.e., "star with dust" (2.), silicate dust emission present (SE), silicate emission at 12-microns (a), uncertain (:)
