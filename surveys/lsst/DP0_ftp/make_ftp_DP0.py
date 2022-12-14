import healpy as hp
import numpy as np
import astropy.io.fits as fits
from astropy.io.fits import getdata
import glob

def make_ftp_ipix(infile, nside_out, nest=True):
    '''
    '''
    data = getdata(infile)
    ra = data['ra']
    dec = data['dec']
    hpx = hp.ang2pix(nside_out, ra, dec, nest=nest, lonlat=True)
    HPX_UN = np.unique(hpx)
    SIGNAL = np.ones(len(HPX_UN))
    
    col0 = fits.Column(name="HP_PIXEL_NEST_4096", format="J", array=HPX_UN)
    col1 = fits.Column(name="SIGNAL", format="E", array=SIGNAL)

    print(infile.split('/')[-1])
    cols = fits.ColDefs([col0, col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto('./' + infile.split('/')[-1], overwrite=False)

DP0_files = glob.glob('/lustre/t1/cl/lsst/dp0_skinny/DP0/DP0_FULL/healpix/32/*.fits')

for i in DP0_files:
    make_ftp_ipix(i, 4096)
