import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.io import fits
from scipy.signal import savgol_filter

__all__ = ['load_fits_data',
           'spectra_preview',
           'energy_normalization',
           'spectra_smooth',
           'spectra_simulation',
           'extract_spectral_features',
           'extract_stat_features']

def load_fits_data(file_path):
    '''
    Load all data from the FITS file. 

    Parameters
    ----------
    file_path : str
        The path to the FITS file.

    Returns
    -------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    '''
    with fits.open(file_path) as hdulist:
        flux_data = hdulist[0].data
        table_data = hdulist[1].data
        label_list = table_data['label']
        data_array = np.hstack((flux_data, label_list.reshape(-1, 1)))
    return data_array

def spectra_preview(data_array, num_spectra_per_class=2, random_state=None):
    '''
    Plot previews of several randomly selected spectra for the data array.

    Parameters
    ----------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    num_spectra_per_class : int, optional
        The number of spectra to plot for each class (galaxy, quasar, star). Default is 2.
    random_state : int, optional
        Random seed for reproducibility. Default is None.
    
    Returns
    -------
    None
    '''
    if random_state is not None:
        np.random.seed(random_state)
    wavelength = np.linspace(3900, 9000, 3000)
    label_map = {0: 'Galaxy', 1: 'Quasar', 2: 'Star'}
    for label in [0, 1, 2]:
        class_data = data_array[data_array[:, -1] == label]
        selected_indices = np.random.choice(class_data.shape[0], num_spectra_per_class, replace=False)
        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(6,3), dpi=100)
            plt.plot(wavelength, class_data[idx, :-1], color='black', linewidth=0.3)
            plt.text(0.02, 0.92, f'{label_map[label]} {i+1}', fontsize=12, transform=plt.gca().transAxes)
            plt.xlabel('Wavelength [{}]'.format(r'$\mathrm{\AA}$'))
            plt.ylabel('Flux')
            plt.tight_layout()
            plt.show()
    return

def energy_normalization(data_array):
    '''
    Normalize each spectrum based on its total flux (energy). 

    Parameters
    ----------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    
    Returns
    -------
    normalized_data_array : numpy.ndarray
        The new data array with normalized flux. Maintains the same shape as the input data array.
    '''
    flux, label = data_array[:, :-1], data_array[:, -1]
    l2_norms = np.linalg.norm(flux, axis=1, keepdims=True) + 1e-8
    flux_normalized = flux / l2_norms
    normalized_data_array = np.hstack((flux_normalized, label.reshape(-1, 1)))
    return normalized_data_array

def spectra_smooth(data_array, window_length=10, polyorder=3):
    '''
    Smooth each spectrum using a Savitzky-Golay filter. 

    Parameters
    ----------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    window_length : int, optional
        The length of the filter window. Default is 10.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Default is 3.
    
    Returns
    -------
    smoothed_data_array : numpy.ndarray
        The new data array with smoothed flux. Maintains the same shape as the input data array.
    '''
    flux, label = data_array[:, :-1], data_array[:, -1]
    flux_smooth = np.zeros_like(flux)
    for i in range(flux.shape[0]):
        flux_smooth[i] = savgol_filter(flux[i], window_length=window_length, polyorder=polyorder)
    smoothed_data_array = np.hstack((flux_smooth, label.reshape(-1, 1)))
    return smoothed_data_array

def spectra_simulation(data_array, catagary_to_sim, number_to_sim, original_SNR=100):
    '''
    Data argumentation: generate new spectra by adding random gaussian noise 
    to existing spectra of a specified category.

    Parameters
    ----------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    catagary_to_sim : int
        The category of spectra to simulate (0: galaxy, 1: quasar, 2: star).
    number_to_sim : int
        The number of new spectra to generate.
    original_SNR : float, optional
        The original signal-to-noise ratio of the spectra. Default is 100.

    Returns
    -------
    simulated_data_array : numpy.ndarray
        The array for generated spectra. Maintains the same shape as the input data array.
    '''
    flux, label = data_array[:, :-1], data_array[:, -1]
    flux_used = flux[label == catagary_to_sim]
    simulated_data_array = np.empty((number_to_sim, data_array.shape[1]))
    for i in range(number_to_sim): 
        selected_index = np.random.choice(flux_used.shape[0], 1, replace=False)
        flux_selected = flux_used[selected_index][0]
        flux_simulated = flux_selected + np.random.normal(loc=0, 
                                                          scale=np.mean(np.abs(flux_selected)) / original_SNR,
                                                          size=flux_selected.shape)
        flux_simulated = np.append(flux_simulated, catagary_to_sim)
        simulated_data_array[i] = flux_simulated
    return simulated_data_array

def extract_spectral_features(data_array, line_centers, window=3):
    '''
    Extract spectral features from specific lines. Return the mean flux 
    within a window around each line center. Frequencies are in Angstroms.

    Parameters
    ----------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    line_centers : list of float
        The central frequencies of the spectral lines to extract features from.
    window : float, optional
        The window size around each line center to consider for feature extraction. Default is 3.
    
    Returns
    -------
    spectral_features : numpy.ndarray
        The extracted spectral features. Has a shape of (data_array.shape[0], len(line_centers)).
    '''
    wavelength = np.linspace(3900, 9000, 3000)
    flux = data_array[:, :-1]
    spectral_features = []
    for pos in line_centers:
        indices = np.where((wavelength >= pos - window) & (wavelength <= pos + window))[0]
        feature_flux = np.mean(flux[:, indices], axis=1).reshape(-1, 1)
        spectral_features.append(feature_flux)
    return np.hstack(spectral_features)

def extract_stat_features(data_array, n=100):
    '''
    Extract basic statistical features from the spectra. The function divides the spectra into n blocks 
    and calculates the mean, variance, argmax, argmin, max, and min for each block.

    Parameters
    ----------
    data_array : numpy.ndarray
        The data array containing flux and labels (last column). Each row corresponds to one spectrum.
    n : int, optional
        The number of blocks to divide each spectrum into. Default is 100. 
    
    Returns
    -------
    stat_features : numpy.ndarray
        The extracted statistical features. Has a shape of (data_array.shape[0], n * 6).
    '''
    flux = data_array[:, :-1]
    step = math.ceil(flux.shape[1] / n)
    means, vars = np.zeros((flux.shape[0], n)), np.zeros((flux.shape[0], n))
    argmaxs, argmins = np.zeros((flux.shape[0], n)), np.zeros((flux.shape[0], n))
    maxs, mins = np.zeros((flux.shape[0], n)), np.zeros((flux.shape[0], n))
    for i in range(n):
        start = i * step
        stop = min((i + 1) * step, flux.shape[1])
        block = flux[:, start:stop]
        means[:, i], vars[:, i] = np.mean(block, axis=1), np.var(block, axis=1)
        argmaxs[:, i], argmins[:, i] = np.argmax(block, axis=1) + start, np.argmin(block, axis=1) + start
        maxs[:, i], mins[:, i] = np.max(block, axis=1), np.min(block, axis=1)
    return np.hstack((means, vars, argmaxs, argmins, maxs, mins))
