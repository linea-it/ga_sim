import healpy as hp
import numpy as np
import astropy.io.fits as fits
from astropy.io.fits import getdata
import glob

def make_ftp_ipix(ii, ipix, cov):
    '''
    '''
    col0 = fits.Column(name="HP_PIXEL_NEST_4096", format="J", array=ipix)
    col1 = fits.Column(name="SIGNAL", format="E", array=cov)

    cols = fits.ColDefs([col0, col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto('./' + str(int(ii)) + '.fits', overwrite=True)

files = 'vac_ga.footprint_6733_19259.fits'
data = getdata(files)
px = data['pixel']
ra, dec = hp.pix2ang(4096, px, nest=True, lonlat=True)
cov_g = data['detfrac_g']
ipix_32 = hp.ang2pix(32, ra, dec, nest=True, lonlat=True)
ipix_32_un = np.unique(ipix_32)

for i in ipix_32_un:
    make_ftp_ipix(i, px[ipix_32 == i], cov_g[ipix_32 == i])
