# -*- coding: utf-8 -*-
import os
import astropy.io.fits as fits
import glob
from astropy.io.fits import getdata
from pathlib import Path

files = glob.glob('/lustre/t1/cl/des/Y6A2_COADD/Y6A2_GOLD/healpix/32/*.fits')

for f in files:
    data = getdata(f)
    RA = data['RA']
    DEC = data['DEC']
    EBV = data['EBV']
    MAG_G = data['SOF_BDF_MAG_G_CORRECTED']
    MAGERR_G = np.array(table['SOF_BDF_MAG_ERR_G']
    MAG_R = data['SOF_BDF_MAG_R_CORRECTED']
    MAGERR_R = data['SOF_BDF_MAG_ERR_R']

    selection_1 = (data['SPREAD_MODEL_I'] + 3. * data['SPREADERR_MODEL_I']) > 0.005
    selection_2 = (data['SPREAD_MODEL_I'] + 1. * data['SPREADERR_MODEL_I']) > 0.003
    selection_3 = (data['SPREAD_MODEL_I'] - 1. * data['SPREADERR_MODEL_I']) > 0.002
    ext_coadd = selection_1.astype(int) + selection_2.astype(int) + selection_3.astype(int)
    
    MAG_G -= 3.186 * EBV
    MAG_R -= 2.140 * EBV

    cond = (np.abs(MAG_G) < mmax) & (np.abs(MAG_G) > mmin) & (
        MAG_G-MAG_R > cmin) & (MAG_G-MAG_R < cmax) & (ext_coadd < 2)

    RA = RA[cond]
    DEC = DEC[cond]
    MAG_G = MAG_G[cond]
    MAG_R = MAG_R[cond]
    MAGERR_G = MAGERR_G[cond]
    MAGERR_R = MAGERR_R[cond]

    output_file = f.split('/')[-1]
    col1 = fits.Column(name='RA', format='D', array=RA)
    col2 = fits.Column(name='DEC', format='D', array=DEC)
    col3 = fits.Column(name='MAG_G', format='E', array=MAG_G)
    col4 = fits.Column(name='MAG_R', format='E', array=MAG_R)
    col5 = fits.Column(name='MAGERR_G', format='E', array=MAGERR_G)
    col6 = fits.Column(name='MAGERR_R', format='E', array=MAGERR_R)

    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(output_file, overwrite=True)


