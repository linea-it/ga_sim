# ## ga_sim

# This jn is intended to create simulations of dwarf galaxies and globular clusters using as field stars the catalog of DES. These simulations will be later copied to gawa jn, a pipeline to detect stellar systems and fields stars in the same catalog. In principle, this pipeline read a table in data base with g and r (or bluer and redder) magnitudes, correct the stars for extinction in each band, and randomize its positions in RA and DEC. This is done to avoid stellar systems in the region simulated (and later inspected by the code of detection). The star clusters are inserted later, centered in each HP pixel with specific nside.
# 
# All the parameters are smarized in a JSON file in the same folder as this notebook.
# 
# To complete all the steps you just have to run all the cells below in sequence.

# Firstly, follow the instructions in the README file to install an enviroment with the packages (using terminal). Do it only once.
# 
# Following, activate the env in the terminal. In the notebook, restart the kernel and select the env created. Now, you are able to properly run the cells bellow.

# Run this jupyter notebook in the LIneA env with the following command:
# <br>
# `jupyter nbconvert --execute --to html --EmbedImagesPreprocessor.embed_images=True ga_sim.ipynb`
# <br>
# and after the command has finished, run the following cell:
# <br>
# `cp ga_sim.html ~/public_html/gawa_processes/NNNNN/simulations/`
# <br>
# where NNNNN is the process number.

# In[1]:


import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import astropy.io.fits as fits
from astropy.io.fits import getdata
from pathlib import Path
import healpy as hp
import sqlalchemy, json, os, sys, glob, parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
from time import sleep
from tqdm import tqdm
import condor
import sys
sys.path.append('/home/adriano.pieres/ga_sim/ga_sim')
sys.path.append('/home/adriano.pieres/ga_sim')

from ga_sim import (
    make_footprint,
    faker,
    join_cat,
    write_sim_clus_features,
    download_iso,
    read_cat,
    gen_clus_file,
    read_error,
    clus_file_results,
    join_cats_clean,
    split_files,
    clean_input_cat,
    clean_input_cat_dist,
    export_results
)
parsl.clear()
parsl.load(condor.get_config('htcondor'))
parsl.set_stream_logger()
# parsl.set_file_logger('log_file.txt', level=logging.DEBUG)

# Below, load the JSON config file (change any parameter in case you want) and create folders for results.
# 
# Load the maps for reddening.

# In[2]:


# Main settings:
confg = "ga_sim.json"

# read config file
with open(confg) as fstream:
    param = json.load(fstream)

# Diretório para os resultados
os.system("mkdir -p " + param['results_path'])

# Reading reddening files
hdu_ngp = fits.open(param['red_maps_path'] + "/SFD_dust_4096_ngp.fits", memmap=True)
ngp = hdu_ngp[0].data

hdu_sgp = fits.open(param['red_maps_path'] + "/SFD_dust_4096_sgp.fits", memmap=True)
sgp = hdu_sgp[0].data


# Downloading the isochrone table with the last improvements from PARSEC.
# Printing age and metalicity of isochrone downloaded. Try one more time in case of empty file. Sometimes there are problems with connection.

# In[7]:


download_iso(param['padova_version_code'], param['survey'], 0.0152 * (10 ** param['MH_simulation']),
             param['age_simulation'], param['av_simulation'], param['file_iso'], 5)


# Checking age and metalicity of the isochrone. 
# 
# mean_mass is the average mass in the range of magnitudes. This is important to calculate a useful range of masses and
# avoid wast time simulating a large number of low-mass stars that are not visible in the range of magnitudes in simulation.

# In[3]:


# Reading [M/H], log_age, mini, g
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


# Creating the footprint map as a flat map. TODO: Use astropy_healpix to improve the ftp map (better sample of data).

# In[4]:


hpx_ftp = make_footprint(param['ra_min'], param['ra_max'], param['dec_min'], param['dec_max'],
                         param['nside_ftp'], output_path=param['results_path'])
print(len(hpx_ftp))


# Reading the catalog from DB, writing it as a fits file and returning as positions, magnitudes free of extinction (top of Galaxy, in order to agree to the stellar models) and its errors.
# 
# The stars are also filtered in by magnitude and color ranges.
# 
# Avoid read from the DB many times in the case the same catalog will be used multiple times.
# 
# DP0 data: tablename for DP0: dp0.dp0_full_des
# extendedness: 0 for stars, 1 for galaxies
# 
# |       Column         |       Type       |
# | :------------------- | ---------------: |
# | coadd_objects_id     | bigint           |
# | z_true               | real             |
# | ra                   | double precision |
# | dec                  | double precision |
# | extendedness         | double precision |
# | mag_u                | double precision |
# | mag_g                | double precision |
# | mag_r                | double precision |
# | mag_i                | double precision |
# | mag_z                | double precision |
# | mag_y                | double precision |
# | magerr_u             | double precision |
# | magerr_g             | double precision |
# | magerr_r             | double precision |
# | magerr_i             | double precision |
# | magerr_z             | double precision |
# | magerr_y             | double precision |
# | host_galaxy          | bigint           |
# | is_variable          | integer          |
# | is_pointsource       | integer          |
# | truth_type           | bigint           |
# | match_sep            | double precision |
# | cosmodc2_hp          | bigint           |
# | cosmodc2_id          | bigint           |
# | is_good_match        | boolean          |
# | is_unique_truth_entry| boolean          |

# In[ ]:

print('Now reading catalog.')

RA, DEC, MAG_G, MAGERR_G, MAG_R, MAGERR_R = read_cat(
    param['vac_ga'], param['ra_min'], param['ra_max'], param['dec_min'], param['dec_max'],
    param['mmin'], param['mmax'], param['cmin'], param['cmax'],
    param['survey'] + "_derred.fits", 1.17450, 0.86666, ngp, sgp, param['results_path'],
    param['results_path'] + "/ftp_4096_nest.fits", param['nside3'], param['nside_ftp'])


# ## Simulation of dwarf galaxies and globular clusters
# 
# In fact, the dwarf galaxies and globular clusters are very similar in terms of stellar populations and stellar density profiles.
# Dwarf galaxies have a half-light radius a bit larger than globular clusters (given the amount of dark matter), keeping constant absolute magnitude. The cell below writes the features of a sample of simulated stellar clustesr given the range
# of positions, visible masses, distances, exponential radius, ellipticity, positional angle and removing clusters too close to edge of simulated region (border_extract, in degrees). All these pars are read from JSON file.

# Generating the properties of clusters based on properties stated above. Writting to file 'objects.dat'.

# In[ ]:

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


# ## Dist stars
# Reading data from magnitude and errors.
# 
# 

# In[ ]:


mag1_, err1_, err2_ = read_error(param['file_error'], 0.000, 0.000)


# Now simulating the clusters.
# 
# Code in the cell below simulates stars using a Kroupa or Salpeter as Initial Mass Function (reading initial mass from models). Given the exponential radius, ellipticity and positional angle the stars are distributed into the sky. 
# 
# Final table contents the position in sky (RA and DEC), magnitudes + its errors, standard deviation of expected errors and initial masses. 

# In[ ]:

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

# Diretório dos arquivo _clus.dat gerados pela faker.
fake_clus_path = param['results_path'] + '/fake_clus'

for i in range(len(hp_sample_un)):
    # Estimating the number of stars in cmd dividing mass by mean mass
    N_stars_cmd = int(mass[i] / mean_mass)
        # os.register_at_fork(after_in_child=lambda: _get_font.cache_clear())
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

# Now, the cell below joins catalogs of simulated clusters and field stars into a single one (optional), and it estimates signal-to-noise ratio from simulations.

# In[ ]:
print('Now joining catalog.')

mockcat = join_cat(
    param['ra_min'],
    param['ra_max'],
    param['dec_min'],
    param['dec_max'],
    hp_sample_un,
    param['survey'],
    RA,
    DEC,
    MAG_G,
    MAG_R,
    MAGERR_G,
    MAGERR_R,
    param['nside_ini'],
    param['mmax'],
    param['mmin'],
    param['cmin'],
    param['cmax'],
    input_path=fake_clus_path,
    output_path=param['results_path'])


# If necessary, split the catalog with simulated clusters and field stars into many files according HP schema.
# 
# This is important in case of large catalogs, avoiding create a large single file.

# In[ ]:

print('Now spliting catalog')

os.makedirs(param['hpx_cats_path'], exist_ok=True)

os.makedirs(param['hpx_cats_clean_path'], exist_ok=True)

ipix_cats = split_files(mockcat, 'ra', 'dec',
                        param['nside_ini'], param['hpx_cats_path'])


# ## Removing stars in crowded fields
# 
# In large surveys, usually objects very close to each other are blended into a single object. In principle, the software used to detect sources is SExtractor, which parameter deblend is set to blend sources very close to each other.
# 
# In order to reproduce this observational bias, cells below remove both stars if they are closer than an specific distance.
# 
# To do that, the approach below (or the function on that) read catalogs from ipixels (HealPixels).
# To each star the distance to all sources are calculated. If the second minimum distance (the first one is zero, since it is the iteration of the stars with itself) is less than the distance defined as a parameter of the function, the star is not listed in the filtered catalog.
# The function runs in parallel, in order to run faster using all the cores of node.
# 
# Firstly, setting the string to read position of stars. After filtering stars in HealPixels, join all the HP into a single catalog called final cat.

# In[ ]:

print('This is the most time consuming part: cleaning the stars from crowding.')

results_from_clear = []

@python_app
def clean_input_cat_dist_app(i, param):

    from ga_sim import clean_input_cat_dist
    
    aaaa = clean_input_cat_dist(param['hpx_cats_clean_path'], i, param['ra_str'], param['dec_str'], param['min_dist_arcsec'], 0.01)
    return aaaa


for i in ipix_cats:
    print(i)
    results_from_clear.append(clean_input_cat_dist_app(i, param))

outputs = [r.result() for r in results_from_clear]

print('Total of {:d} pixels were cleaned from crowding fields.'.format(np.sum(outputs)))
# In[ ]:

ipix_clean_cats = [i.replace(param['hpx_cats_path'], param['hpx_cats_clean_path']) for i in ipix_cats]

#import time

#while len(glob.glob(param['hpx_cats_clean_path']+ '/*')) < len(ipix_clean_cats):
#    time.sleep(3)

join_cats_clean(ipix_clean_cats,
                param['final_cat'], param['ra_str'], param['dec_str'])

print('Almost done.')
# In[ ]:


sim_clus_feat = write_sim_clus_features(
    mockcat,
    param['final_cat'],
    hp_sample_un,
    param['nside_ini'],
    mM,
    param['results_path'],
    param['file_error'],
    param['file_mask'],
    param['snr_inner_circle_arcmin'],
    param['snr_rin_annulus_arcmin'],
    param['snr_rout_annulus_arcmin']
)

clus_file_results(param['star_clusters_simulated'], sim_clus_feat, param['results_path'] + '/objects.dat')

os.system('jupyter nbconvert --execute --to html --EmbedImagesPreprocessor.embed_images=True plots_sim.ipynb')

export_results(param['export_path'], param['results_path'], param['copy_html_path'])

