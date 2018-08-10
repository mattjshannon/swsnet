===============================
swsnet
===============================

.. image:: https://img.shields.io/travis/mattjshannon/swsnet.svg
        :target: https://travis-ci.org/mattjshannon/swsnet

.. image:: https://img.shields.io/pypi/v/swsnet.svg
        :target: https://pypi.python.org/pypi/swsnet


Applying neural networks to the Sloan SWS astronomical dataset.

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://mattjshannon.github.io/swsnet.

Premise
--------

* Predict labels for the larger CASSIS dataset using the SWS dataset for training/validation.

Models
------

Models are presented in Jupyter notebooks (see folder ipy_notebooks). Primary focus right now is a neural network.
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
