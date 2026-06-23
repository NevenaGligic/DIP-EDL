# if running on a server without display, uncomment the following line
# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def get_spectrum(file, index):
    '''
    Read a single spectrum from the FITS file. 

    Parameters
    ----------
    file : str
        The path to the FITS file.
    index : int
        The index of the spectrum to read. Starting from 0.
    
    Returns
    -------
    wavelength : numpy.ndarray
        The wavelength array of the spectrum.
    flux : numpy.ndarray
        The flux array of the spectrum.
    label : int
        0 - 'Galaxy', 1 - 'Quasar', 2 - 'Star'
    '''
    with fits.open(file) as hdulist:
        flux = hdulist[0].data[index]
        label = hdulist[1].data['label'][index]
    wavelength = np.linspace(3900, 9000, 3000)
    return wavelength, flux, label

if __name__ == '__main__':

    wavelength, flux, label = get_spectrum(file='train_data_01.fits', index=6500)
    label_map = {0:'Galaxy', 1:'Quasar', 2:'Star'}

    plt.figure(figsize=(6,3), dpi=100)
    plt.plot(wavelength, flux, color='black', linewidth=0.3)
    plt.text(0.02, 0.92, f'{label_map[label]}', fontsize=12, transform=plt.gca().transAxes)
    plt.xlabel('Wavelength [{}]'.format(r'$\mathrm{\AA}$'))
    plt.ylabel('Flux')
    plt.tight_layout()
    plt.savefig('spectrum_preview.png')
    plt.show()
