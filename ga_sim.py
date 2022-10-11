from ga_sim import (
    make_footprint,
    faker,
    join_cat,
    write_sim_clus_features,
    download_iso,
    filter_ipix_stars,
    gen_clus_file,
    read_error,
    clus_file_results,
    join_cats_clean,
    split_files,
    clean_input_cat_dist,
    export_results,
    select_ipix,
    resize_ipix_cats,
    estimation_area,
    join_sim_field_stars
)

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import astropy.io.fits as fits
from astropy.io.fits import getdata
from pathlib import Path
import healpy as hp
import sqlalchemy
import json
import os
import sys
import glob
import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
from time import sleep
from tqdm import tqdm
import condor
import sys

# Loading config and files, creating folders
parsl.clear()
parsl.load(condor.get_config('htcondor'))
parsl.set_stream_logger()

confg = "ga_sim.json"

with open(confg) as fstream:
    param = json.load(fstream)

os.makedirs(param['results_path'], exist_ok=True)
os.makedirs(param['ftp_path'], exist_ok=True)
os.makedirs(param['hpx_cats_clus_field'], exist_ok=True)
os.makedirs(param['hpx_cats_path'], exist_ok=True)
os.makedirs(param['hpx_cats_clean_path'], exist_ok=True)
os.makedirs(param['hpx_cats_filt_path'], exist_ok=True)

hdu_ngp = fits.open(param['red_maps_path'] +
                    "/SFD_dust_4096_ngp.fits", memmap=True)
ngp = hdu_ngp[0].data

hdu_sgp = fits.open(param['red_maps_path'] +
                    "/SFD_dust_4096_sgp.fits", memmap=True)
sgp = hdu_sgp[0].data

# Downloading isochrone and printing some information

download_iso(param['padova_version_code'], param['survey'], 0.0152 * (10 ** param['MH_simulation']),
             param['age_simulation'], param['av_simulation'], param['file_iso'], 5)


iso_info = np.loadtxt(param['file_iso'], usecols=(1, 2, 3, 26), unpack=True)
FeH_iso = iso_info[0][0]
logAge_iso = iso_info[1][0]
m_ini_iso = iso_info[2]
g_iso = iso_info[3]

print('[Fe/H]={:.2f}, Age={:.2f} Gyr'.format(FeH_iso, 10**(logAge_iso-9)))

mM_mean = (param['mM_max'] + param['mM_min']) / 2.
print(np.max(m_ini_iso[g_iso + mM_mean < param['mmax']]))
mean_mass = (np.min(m_ini_iso[g_iso + mM_mean < param['mmax']]) +
             np.max(m_ini_iso[g_iso + mM_mean < param['mmax']])) / 2.

print('Mean mass (M_sun): {:.2f}'.format(mean_mass))

# Making footprint
hpx_ftp = make_footprint(param)

# Calculating area sampled and defining mode       
area_sampled = estimation_area(param)

if (param['survey'] == 'lsst') and (area_sampled < 300.):
    mode = 'expand'
else:
    mode = 'cutout'

# Selecting input files and filtering by magnitude and color ranges and 
# correcting for extinction
ipix_files = select_ipix(param['nside_infile'], param['ra_min'], param['ra_max'],
                         param['dec_min'], param['dec_max'], True)

@python_app
def filter_ipix_stars_app(ipix, param, ngp, sgp):

    from ga_sim import filter_ipix_stars

    filter_ipix_stars(ipix, param, ngp, sgp)

print('Now reading catalog.')

for i in ipix_files:
    filter_ipix_stars_app(i, param, ngp, sgp)

# Expanding catalog depending on the case
resize_ipix_cats(ipix_files, param, mode, area_sampled)

# Generating features of simulated clusters
print('Now generating cluster file.')

RA_pix, DEC_pix, r_exp, ell, pa, dist, mass, mM, hp_sample_un = gen_clus_file(
    param['ra_min'],
    param['ra_max'],
    param['dec_min'],
    param['dec_max'],
    param['nside_ini'],
    param['border_extract'],
    param['mM_min'],
    param['mM_max'],
    param['log10_rexp_min'],
    param['log10_rexp_max'],
    param['log10_mass_min'],
    param['log10_mass_max'],
    param['ell_min'],
    param['ell_max'],
    param['pa_min'],
    param['pa_max'],
    param['results_path']
)

# Loading photometric errors
mag1_, err1_, err2_ = read_error(param['file_error'], 0.000, 0.000)


# Simulating stellar clusters.
print('Ready to simulate clusters.')

@python_app
def faker_app(N_stars_cmd, frac_bin, IMF_author, x0, y0, rexp, ell_, pa, dist, hpx, param, mag1_, err1_, err2_, output_path, mag_ref_comp,
              comp_mag_ref,
              comp_mag_max):

    from ga_sim import faker

    faker(
        N_stars_cmd,
        frac_bin,
        IMF_author,
        x0,
        y0,
        rexp,
        ell_,
        pa,
        dist,
        hpx,
        param['cmin'],
        param['cmax'],
        param['mmin'],
        param['mmax'],
        mag1_,
        err1_,
        err2_,
        param['file_iso'],
        output_path,
        mag_ref_comp,
        comp_mag_ref,
        comp_mag_max,
    )


fake_clus_path = param['results_path'] + '/fake_clus'

for i in range(len(hp_sample_un)):
    N_stars_cmd = int(mass[i] / mean_mass)
    faker_app(
        N_stars_cmd,
        param['frac_bin'],
        param['IMF_author'],
        RA_pix[i],
        DEC_pix[i],
        r_exp[i],
        ell[i],
        pa[i],
        dist[i],
        hp_sample_un[i],
        param,
        mag1_,
        err1_,
        err2_,
        fake_clus_path,
        param['mag_ref_comp'],
        param['comp_mag_ref'],
        param['comp_mag_max'],
    )

ipix_ini = select_ipix(param['nside_ini'], param['ra_min'], param['ra_max'],
                         param['dec_min'], param['dec_max'], True)

results_join = []

@python_app
def join_sim_field_stars_app(ipix, param):

    from ga_sim import join_sim_field_stars

    aaaa = join_sim_field_stars(ipix, param)

    return aaaa

print('Now starting to join simulations and field stars.')

for i in ipix_ini:
    results_join.append(join_sim_field_stars_app(i, param))

outputs = [r.result() for r in results_join]

print('Total of {:d} pixels were joint from clusters and fields.'.format(
    int(np.sum(outputs))))

ipix_cats = glob.glob(param['hpx_cats_clus_field'] + '/*.fits')

print('This is the most time consuming part: cleaning the stars from crowding.')

results_from_clear = []

@python_app
def clean_input_cat_dist_app(i, param):

    from ga_sim import clean_input_cat_dist

    aaaa = clean_input_cat_dist(param['hpx_cats_clus_field'], param['hpx_cats_clean_path'], i,
                                param['ra_str'], param['dec_str'], param['min_dist_arcsec'], 0.01)
    return aaaa


for i in ipix_cats:
    results_from_clear.append(clean_input_cat_dist_app(i, param))

outputs = [r.result() for r in results_from_clear]

print('Total of {:d} pixels were cleaned from crowding fields.'.format(
    int(np.sum(outputs))))

ipix_clean_cats = [i.replace(
    param['hpx_cats_path'], param['hpx_cats_clean_path']) for i in ipix_cats]

print('Almost done.')

# Solve name of variable
sim_clus_feat = write_sim_clus_features(param, hp_sample_un, mM)

clus_file_results(param['star_clusters_simulated'], sim_clus_feat, param['results_path'] + '/objects.dat')

os.system('jupyter nbconvert --execute --to html --EmbedImagesPreprocessor.embed_images=True plots_sim.ipynb')

export_results(param['export_path'], param['results_path'], param['copy_html_path'])
