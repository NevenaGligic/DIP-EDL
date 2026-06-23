'''
Utility package for the spectra classification.

Includes:
- Data loading and preprocessing (`functions.py`)
- CNN model definition (`CNN_model.py`)
'''

from .functions import (
    load_fits_data,
    spectra_preview,
    energy_normalization,
    spectra_smooth,
    spectra_simulation,
    extract_spectral_features,
    extract_stat_features
)

from .CNN_model import CNN_Model_1D
