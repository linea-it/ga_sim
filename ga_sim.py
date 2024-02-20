from ga_sim import (
    make_footprint,
    write_sim_clus_features,
    download_iso,
    gen_clus_file,
    read_error,
    clus_file_results,
    export_results,
    select_ipix,
    sample_ipix_cat,
    estimation_area
)

import numpy as np
import astropy.io.fits as fits
import json
import os
import glob
import parsl
from parsl.app.app import python_app
import sys
import healpy as hp
import cProfile, pstats
from ga_sim.parsl_config import get_config
import logging

profiler = cProfile.Profile()
profiler.enable()
# Loading config and files, creating folders
parsl.clear()
#parsl_config = get_config(ga_sim_config["executor"])
#parsl.set_stream_logger()

logname = '/lustre/t1/cl/lsst/gawa_project/adriano.pieres/ga_sim/ga_sim.log'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

logging.warning('Packages were imported and parsl configuration was loaded.')

confg = "ga_sim.json"

with open(confg) as fstream:
    param = json.load(fstream)

parsl_config = get_config(param["executor"])
parsl.set_stream_logger()
parsl.load(parsl_config)

try:
    os.system('rm -r results/hpx*')
    os.system('rm results/ftp/*.fits')
    os.system('rm results/objects.dat')
except:
    print('No data to clean.')

os.makedirs(param['results_path'], exist_ok=True)
os.makedirs(param['ftp_path'], exist_ok=True)
os.makedirs(param['hpx_cats_clus_field'], exist_ok=True)
os.makedirs(param['hpx_cats_path'], exist_ok=True)
os.makedirs(param['hpx_cats_clean_path'], exist_ok=True)
os.makedirs(param['hpx_cats_filt_path'], exist_ok=True)
os.makedirs(param['results_path'] + '/fake_clus', exist_ok=True)

logging.warning('Folders were created and download of isochrone will start.')

# Downloading isochrone and printing some information
download_iso(param['padova_version_code'], param['survey'], 0.0152 * (10 ** param['MH_simulation']),
             param['age_simulation'], param['av_simulation'], param["IMF_author"], param['file_iso'], 5)

f = open(param['file_iso'], "r")
cols = f.readlines()
cols = cols[11].split()
cols.pop(0)
n_col_magg = cols.index('gmag')
n_col_magr = cols.index('rmag')
f.close()

logging.warning('Isochrone was downloaded. Starting to create footprint.')

# Making footprint
make_footprint(param)

logging.warning('Footprint was created.')

area_sampled = estimation_area(param)

logging.warning('Starting to select pixels for simulations.')

# Selecting input files and filtering by magnitude and color ranges and
# correcting for extinction
ipix_files = select_ipix(param['nside_infile'], param['ra_min'], param['ra_max'],
                         param['dec_min'], param['dec_max'], True)

@python_app
def filter_ipix_stars_app(ipix, param):

    from ga_sim import filter_ipix_stars

    aaa = filter_ipix_stars(ipix, param)

    return aaa

res = []

for i in ipix_files:
    res.append(filter_ipix_stars_app(i, param))

outputs = [r.result() for r in res]

logging.warning('Pixels to be simulated was selected.')

print('Total of {:d} pixels read and filtered.'.format(int(np.sum(outputs))))

# Expanding catalog depending on the case
print('Area sampled: {:.2f} square degrees'.format(area_sampled))

if param['survey']== 'lsst': ftp_infile_path = "/lustre/t1/cl/lsst/gawa_project/adriano.pieres/ga_sim/surveys/lsst/DP0_ftp"
if param['survey']== 'des': ftp_infile_path = "/lustre/t1/cl/lsst/gawa_project/adriano.pieres/ga_sim/surveys/des/ftp"

files_ftp = glob.glob(param['ftp_path'] + '/*.fits')
files_DP0_ftp = glob.glob(ftp_infile_path + '/' + str(int(param['nside_infile'])) + '/*.fits')

'''
Ver se essa parte do DP0 é realmente necessária.
'''

good_DP0_ftp = []

for ii in files_DP0_ftp:
    data = fits.getdata(ii)
    signal = data['SIGNAL']
    cov_fact_ipix = np.sum(signal) * hp.nside2pixarea(param['nside_ftp'], degrees=True) / hp.nside2pixarea(param['nside_infile'], degrees=True)
    if cov_fact_ipix > param['cov_factor']:
        if int((ii.split('/')[-1]).split('.')[0]) in ipix_files:
            good_DP0_ftp.extend([ii])

ipix_ftp = [i.split('/')[-1] for i in files_ftp]

logging.warning('Pixels with good amount of field stars in survey was selected.')

logging.warning('Sampling pixels with field stars in survey.')

@python_app
def sample_ipix_cat_app(i, good_DP0_ftp, param):
    from ga_sim import sample_ipix_cat
    aaa = sample_ipix_cat(i, good_DP0_ftp, param)

res2 = []

for i in files_ftp:
    res2.append(sample_ipix_cat_app(i, good_DP0_ftp, param))

outputs = [r.result() for r in res2]

logging.warning('Field stars in pixels was created. Now start to generate cluster files.')

# Creating clusters
RA_pix, DEC_pix, r_exp, ell, pa, dist, mM, hp_sample_un, MV = gen_clus_file(
    param)

# Loading photometric errors
mag1_, err1_, err2_ = read_error(param['file_error'] + '/' + param['survey'] + '/errors.dat', 0.000, 0.000)

@python_app
def faker_app(MV, frac_bin, x0, y0, rexp, ell_, pa, mM, hpx, param, mag1_, err1_, err2_, output_path, mag_ref_comp,
              comp_mag_ref, comp_mag_max, n_magg, n_magr):

    from ga_sim import faker

    bbbb = faker(MV, frac_bin, x0, y0, rexp, ell_, pa, mM, hpx, param['cmin'], param['cmax'],
                 param['mmin'], param['mmax'], mag1_, err1_, err2_, param['file_iso'], output_path, mag_ref_comp,
                 comp_mag_ref, comp_mag_max, n_magg, n_magr)
    return bbbb

fake_clus_path = param['results_path'] + '/fake_clus'

results_faker = []

for i in range(len(hp_sample_un)):

    results_faker.append(faker_app(MV[i], param['frac_bin'], RA_pix[i], DEC_pix[i], r_exp[i], ell[i],
              pa[i], mM[i], hp_sample_un[i], param, mag1_, err1_, err2_, fake_clus_path,
              param['mag_ref_comp'], param['comp_mag_ref'], param['comp_mag_max'], n_col_magg, n_col_magr))

outputs = [r.result() for r in results_faker]

logging.warning('Simulated cluster files were created. Now start to join simulated clusters and field stars.')

ipix_ini = glob.glob(param['hpx_cats_path'] + '/*.fits')

results_join = []

@python_app
def join_sim_field_stars_app(ipix, param):

    from ga_sim import join_sim_field_stars

    aaaa = join_sim_field_stars(ipix, param)

    return aaaa

for i in ipix_ini:
    results_join.append(join_sim_field_stars_app(i, param))

outputs = [r.result() for r in results_join]

logging.warning('Simulated clusters and field stars were joined in a single file. Starting to clear files from crowding')

print('Total of {:d} pixels were joint from clusters and fields.'.format(
    int(np.sum(outputs))))

ipix_cats = glob.glob(param['hpx_cats_clus_field'] + '/*.fits')

print('== LEN IPIX CATS == ', len(ipix_cats))
print('This is the most time consuming part: cleaning the stars from crowding.')

results_from_clear = []

@python_app
def clean_input_cat_dist_app(iiii, param):

    from ga_sim import clean_input_cat_dist

    aaaa = clean_input_cat_dist(param['hpx_cats_clean_path'], iiii, param['ra_str'],
                                param['dec_str'], param['min_dist_arcsec'], 0.01, iiii + '.log')
    return aaaa

for aa in ipix_cats:
    results_from_clear.append(clean_input_cat_dist_app(aa, param))

outputs = [r.result() for r in results_from_clear]

logging.warning('Joint cluster and field stars were cleared from crowding. Finishing the simulation.')

print('Total of {:d} pixels were cleaned from crowding fields.'.format(
    int(np.sum(outputs))))

ipix_clean_cats = [i.replace(
    param['hpx_cats_path'], param['hpx_cats_clean_path']) for i in ipix_cats]

print('Almost done.')

# Solve name of variable
sim_clus_feat = write_sim_clus_features(param, hp_sample_un, mM)

clus_file_results(param['star_clusters_simulated'],
                  sim_clus_feat, param['results_path'] + '/objects.dat')

# os.system('jupyter nbconvert --execute --to html --EmbedImagesPreprocessor.embed_images=True plots_sim.ipynb')
profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
stats = pstats.Stats(profiler)
stats.dump_stats(param['results_path'] + '/cProfile_data.txt')
logging.warning('Start to export results.')

export_results(param['export_path'], param['results_path']) # param['copy_html_path'])

logging.warning('Results were exported and code will close.')

