# -*- coding: utf-8 -*-
import os
import astropy.coordinates as coord
import astropy.io.fits as fits
import healpy as hp
import matplotlib.path as mpath
import numpy as np
import sqlalchemy
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io.fits import getdata
from scipy.stats import expon
from pathlib import Path
from itertools import compress

# @python_app


def king_prof(rc, rt, N_stars):
    """This function distributes stars in a King profile, the number-density profile
    most applied to globular clusters. The final array is the stars distributes along
    projected radii.
    Reference:
    https://www.aanda.org/articles/aa/full/2008/03/aa8616-07/aa8616-07.html

    Parameters
    ----------
    rc : float
        Core radius, not specified units.
    rt : float
        Spherical tidal radius.
    N_stars : int
        Total star counts.
    max_length : float, optional
        Maximum length of , by default 1.

    Returns
    -------
    _type_
        _description_
    """
    rad_star = []
    radius = np.linspace(0, 1., 1000)
    rc /= rt
    peak = np.max(2 * np.pi * radius * ((1 / np.sqrt(1 + (radius / rc)
                  ** 2)) - (1 / np.sqrt(1 + (1. / rc) ** 2))) ** 2)
    while len(rad_star) < N_stars:
        rd_nmb = np.random.rand(1, 2)
        rad = rd_nmb[0][0]
        hei = rd_nmb[0][1] * peak
        if hei <= 2 * np.pi * rad * ((1 / np.sqrt(1 + (rad / rc) ** 2)) -
                                     (1 / np.sqrt(1 + (1. / rc) ** 2))) ** 2:
            rad_star.append(rad)
    return np.multiply(rad_star, rt)


def clean_input_cat_dist(file_name, ra_str, dec_str, max_dist_arcsec):
    """ This function removes stars closer than max_dist_arcsec. That is specially significant to
    stellar clusters, where the stellar crowding in images creates a single
    object in cluster's center, but many star in the periphery.

    Parameters
    ----------
    file_name : str
        Name of input file.
    ra_str : str
        Label for RA coordinate.
    dec_str : str
        Label for DEC coordinate.
    max_dist_arcsec : float
        Stars closer than this par will be removed. Units: arcsec.
    """

    output_file = file_name.split('.')[0] + '_clean_dist.fits'

    data = getdata(file_name)
    t = Table.read(file_name)
    label_columns = t.colnames
    t_format = []
    for i in label_columns:
        t_format.append(t[i].info.dtype)

    clean_idx = []

    RA = data[ra_str]
    DEC = data[dec_str]

    for i, j in enumerate(RA):
        length = 0.1
        cond = (RA < j + length) & (RA > j - length) & (DEC <
                                                        DEC[i] + length) & (DEC > DEC[i] - length)
        RA_ = RA[cond]
        DEC_ = DEC[cond]
        dist = dist_ang(RA_, DEC_, j, DEC[i])
        if (sorted(set(dist))[1] > max_dist_arcsec / 3600):
            clean_idx.append(i)

    data_clean = np.array([data[:][i] for i in clean_idx])

    col = [i for i in range(len(label_columns))]

    for i, j in enumerate(label_columns):
        col[i] = fits.Column(
            name=label_columns[i], format=t_format[i], array=data_clean[:, i])
    cols = fits.ColDefs([col[i] for i in range(len(label_columns))])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    tbhdu.writeto(output_file, overwrite=True)


def exp_prof(rexp, N_stars, max_length=10):
    """This function distributes stars radially following an exponential
    density profile. This function does not distribute stars into 3D,
    only in 1-D, which is complemented in additional code.

    Parameters
    ----------
    rexp : float
        Scale radius of cluster, in degrees or parsecs.
    N_stars : int
        Total number counts.
    max_length : float
        Maximum distance of star in scale radius.

    Returns
    -------
    array
        Radius of stars following exponential density profile.
    """
    rad_star = []
    while len(rad_star) < N_stars:
        rd_nmb = np.random.rand(1, 2)
        rad = rd_nmb[0][0] * max_length * rexp
        hei = rd_nmb[0][1] * 2 * np.pi * rexp / np.e
        if hei <= np.exp(-rad / rexp) * 2 * np.pi * rad:
            rad_star.append(rad)
    return np.asarray(rad_star)


def join_cats_clean(ipix_cats, output_file, ra_str, dec_str):
    """This function writes a single Fits file catalog gathering
    data read in all the ipix_cats.

    Parameters
    ----------
    ipix_cats : list
        List of ipix FITS files catalogs.
    output_file : str
        File name with all data from ipix cats.
    ra_str : str
        Label of RA coordinate.
    dec_str : str
        Label of DEC coordinate.
    """
    t = Table.read(ipix_cats[0])
    label_columns = t.colnames
    t_format = []
    for i in label_columns:
        t_format.append(t[i].info.dtype)

    for i, j in enumerate(ipix_cats):
        if i == 0:
            data = getdata(j)
            all_data = data
        else:
            data = getdata(j)
            all_data = np.concatenate((all_data, data))

    col = [i for i in range(len(label_columns))]

    for i, j in enumerate(label_columns):
        col[i] = fits.Column(
            name=j, format=t_format[i], array=all_data[label_columns[i]])
    cols = fits.ColDefs([col[i] for i in range(len(label_columns))])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(output_file, overwrite=True)


def split_files(in_file, ra_str, dec_str, nside, path):
    """This function split a main file into small catalogs, based on
    HealPix ipix files, with nside=nside.

    Parameters
    ----------
    in_file : str
        Input file.
    ra_str : str
        Label of RA coordinate.
    dec_str : str
        Label of DEC coordinate.
    nside : int
        Nside to split small catalogs.
    path : str
        Path to output files.

    Returns
    -------
    list
        Name and path of output files.
    """

    # os.system('mkdir -p ' + path)

    data = getdata(in_file)
    t = Table.read(in_file)
    label_columns = t.colnames
    t_format = []
    for i in label_columns:
        t_format.append(t[i].info.dtype)
    HPX = hp.ang2pix(nside, data[ra_str],
                     data[dec_str], nest=True, lonlat=True)

    HPX_un = np.unique(HPX)

    for j in HPX_un:
        cond = (HPX == j)
        data_ = data[cond]
        col = [i for i in range(len(label_columns))]

        for i in range(len(label_columns)):
            col[i] = fits.Column(
                name=label_columns[i], format=t_format[i], array=data_[label_columns[i]])
        cols = fits.ColDefs([col[i] for i in range(len(label_columns))])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(path + '/' + str(j) + '.fits', overwrite=True)

    return [(path + '/' + str(i) + '.fits') for i in HPX_un]


# @python_app
def clean_input_cat(file_name, ra_str, dec_str, nside):
    """ This function removes all the stars that resides in the same ipix with
    nside = nside. This is done to simulate the features of real catalogs based on
    detections from SExtractor, where objects very close to each other are
    interpreted as a single object. That is specially significant to
    stellar clusters, where the stellar crowding in images creates a single
    object in cluster's center, but many star in the periphery.
    This function calculates the counts in each ipix with nside (usually > 2 ** 15)
    and removes ALL of stars that resides in the same pixel.

    Parameters
    ----------
    file_name : str
        Name of input file.
    ra_str : str
        Label for RA coordinate.
    dec_str : str
        Label for DEC coordinate.
    nside : int
        Nside to be populated.
    """

    output_file = file_name.split('.')[0] + '_clean.fits'

    data = getdata(file_name)
    t = Table.read(file_name)
    label_columns = t.colnames
    t_format = []
    for i in label_columns:
        t_format.append(t[i].info.dtype)
    HPX = hp.ang2pix(nside, data[ra_str],
                     data[dec_str], nest=True, lonlat=True)

    HPX_idx_sort = np.argsort(HPX)

    HPX_sort = [HPX[i] for i in HPX_idx_sort]
    data_sort = data[HPX_idx_sort]

    a, HPX_idx = np.unique(HPX_sort, return_inverse=True)

    # original order not preserved!
    HPX_un, HPX_counts = np.unique(HPX_sort, return_counts=True)

    HPX_single_star_pix = [
        j for i, j in enumerate(HPX_un) if HPX_counts[i] < 2]

    data_clean = np.array([data_sort[:][i] for i, j in enumerate(HPX_idx) if
                           HPX_un[j] in HPX_single_star_pix])

    col = [i for i in range(len(label_columns))]

    for i, j in enumerate(label_columns):
        col[i] = fits.Column(
            name=label_columns[i], format=t_format[i], array=data_clean[:, i])
    cols = fits.ColDefs([col[i] for i in range(len(label_columns))])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    tbhdu.writeto(output_file, overwrite=True)


def clus_file_results(results_path, out_file, sim_clus_feat, objects_filepath):
    """This function uses the join command to join a file with initial features
    of the simulated clusters to a file with number of stars and absolute
    magnitude since the last file is created after the simulations.

    Parameters
    ----------
    results_path : str
        Path to final results file.
    out_file : str
        Name of final results file.
    sim_clus_feat : str
        Name of file with initial features of clusters.
    objects_filepath : str
        Name of file with n_stars and absolute magnitude.
    """
    star_clusters_simulated = Path(results_path, out_file)
    os.system('join --nocheck-order %s %s > %s' %
              (sim_clus_feat, objects_filepath, star_clusters_simulated))


def read_error(infile, to_add_mag1, to_add_mag2):
    """This function reads the photometric error of survey.
    Values are added to simulate realistic photometric errors.

    Parameters
    ----------
    infile : str
        Name of file with errors.
    to_add_mag1 : float
        Magnitude error to be added to mag1.
    to_add_mag2 : float
        Magnitude error to be added to mag2.

    Returns
    -------
    list
        List of magnitudes and errors in band 1 and 2.
    """
    mag1_, err1_, err2_ = np.loadtxt(infile, usecols=(0, 1, 2), unpack=True)
    err1_ += to_add_mag1
    err2_ += to_add_mag2
    return mag1_, err1_, err2_


def gen_clus_file(ra_min, ra_max, dec_min, dec_max, nside_ini, border_extract,
                  mM_min, mM_max, log10_rexp_min, log10_rexp_max, log10_mass_min,
                  log10_mass_max, ell_min, ell_max, pa_min, pa_max, results_path):
    """This function generates the file with features of simulated clusters,
    based on numerical simulations within the range of parameters.

    Parameters
    ----------
    ra_min : float
        Minimum value of RA, in degrees.
    ra_max : float
        Maximum value of RA, in degrees.
    dec_min : _type_
        Minimum value of DEC, in degrees.
    dec_max : _type_
        Maximum value of DEC, in degrees.
    nside_ini : int
        Nside where the clusters simulated will reside (in the center of ipix).
    border_extract : float
        The angular border (in degrees) where the clusters will not be simulated.
    mM_min : float
        Minimum value for distance modulus (in magnitudes).
    mM_max : _type_
        Maximum value for distance modulus (in magnitudes).
    log10_rexp_min : float
        Minimum value for log_10 of exponential radius, in parsecs.
    log10_rexp_max : float
        Maximum value for log_10 of exponential radius, in parsecs.
    log10_mass_min : float
        Minimum value for log_10 of visible mass (mass of stars that are in
        the range of magnitudes), in Msun.
    log10_mass_max : float
        Maximum value for log_10 of visible mass (mass of stars that are in
        the range of magnitudes), in Msun.
    ell_min : float
        Minimum value for ellipticity (b-a)/a.
    ell_max : float
        Maximum value for ellipticity (b-a)/a.
    pa_min : float
        Minimum value for positional angle (oriented to ), in degrees.
    pa_max : float
        Maximum value for positional angle (oriented to ), in degrees.
    results_path : str
        Path where the final file is written.

    Returns
    -------
    list
        RA_pix, DEC_pix, r_exp, ell, pa, dist, mass, mM, hp_sample_un
        Position of the center of ipix, exponential radius in parsecs,
        ellipticity. positional angle, distance in parsecs, mass in 
        Sun masses, distance modulus, and list of ipix sampled.
    """
    cell_area = hp.nside2pixarea(nside_ini, degrees=True)

    area = (
        (dec_max - dec_min)
        * np.cos(np.deg2rad((ra_max + ra_min) / 2.0))
        * (ra_max - ra_min)
    )

    vertices = hp.ang2vec(
        [
            ra_min + border_extract,
            ra_max - border_extract,
            ra_max - border_extract,
            ra_min + border_extract,
        ],
        [
            dec_min + border_extract,
            dec_min + border_extract,
            dec_max - border_extract,
            dec_max - border_extract,
        ],
        lonlat=True,
    )

    hp_sample_un = hp.query_polygon(
        nside_ini, vertices, inclusive=False, nest=True, buff=None)

    RA_pix, DEC_pix = hp.pix2ang(
        nside_ini, hp_sample_un, nest=True, lonlat=True)

    c = SkyCoord(ra=RA_pix * u.degree, dec=DEC_pix * u.degree, frame='icrs')
    L = c.galactic.l.degree
    B = c.galactic.b.degree

    objects_filepath = Path(results_path, "objects.dat")
    with open(objects_filepath, 'w') as obj_file:

        # Creating random distances, masses, ellipticities and positional
        # angles based on limits required, and printing to file objects.dat.
        mM = mM_min + np.random.rand(len(hp_sample_un)) * (mM_max - mM_min)
        r_exp = 10**(log10_rexp_min * (log10_rexp_max / log10_rexp_min)
                     ** np.random.rand(len(hp_sample_un)))
        mass = 10**(log10_mass_min * (log10_mass_max / log10_mass_min)
                    ** np.random.rand(len(hp_sample_un)))
        dist = 10 ** ((mM/5) + 1)

        ell = ell_min + np.random.rand(len(hp_sample_un)) * (ell_max - ell_min)
        pa = pa_min + np.random.rand(len(hp_sample_un)) * (pa_max - pa_min)

        for i in range(len(hp_sample_un)):
            print('{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
                hp_sample_un[i], L[i], B[i], RA_pix[i], DEC_pix[i], r_exp[i], ell[i], pa[i],
                mass[i], dist[i]), file=obj_file)
    return RA_pix, DEC_pix, r_exp, ell, pa, dist, mass, mM, hp_sample_un


def read_cat(tablename, ra_min, ra_max, dec_min, dec_max, mmin, mmax, cmin, cmax, outfile, AG_AV, AR_AV, ngp, sgp, results_path):
    """Read catalog from LIneA data base.

    Parameters
    ----------
    tablename : str
        Name of FITS file.
    ra_min : float
        Minimum value of RA in degrees.
    ra_max : float
        Maximum value of RA in degrees.
    dec_min : float
        Minimum value of DEC in degrees.
    dec_max : float
        Maximum value of RA in degrees.
    mmin : float
        Minimum value of magnitude.
    mmax : float
        Maximum value of magnitude.
    cmin : float
        Minimum value of color.
    cmax : float
        Maximum value of color.
    outfile : str
       Name of output file.
    AG_AV : float
        Relation of extinction in g band over extinction in V band.
    AR_AV : _type_
        Relation of extinction in r band over extinction in V band.
    ngp : array-like
        Map of reddening for Northern Galactic Hemisphere.
    sgp : array-like
        Map of reddening for Southern Galactic Hemisphere.
    results_path : str
        Path where the results will be written.

    Returns
    -------
    list
        RA, DEC, MAG_G, MAGERR_G, MAG_R, MAGERR_R
        Contents of table where magnitudes are free-extnction (top of Galaxy).
    """
    engine = sqlalchemy.create_engine(
        'postgresql://untrustedprod:untrusted@desdb6.linea.gov.br:5432/prod_gavo')
    conn = engine.connect()

    query = 'select ra, dec, mag_g, magerr_g, mag_r, magerr_r from %s where (ra > %s) and (ra <%s) and (dec > %s) and (dec < %s)' % (
        tablename, ra_min, ra_max, dec_min, dec_max)
    stm = sqlalchemy.sql.text(query)
    stm_get = conn.execute(stm)
    stm_result = stm_get.fetchall()
    table = Table(rows=stm_result, names=(
        'ra', 'dec', 'mag_g', 'magerr_g', 'mag_r', 'magerr_r'))

    RA = np.array(table['ra'])
    DEC = np.array(table['dec'])
    MAG_G = np.array(table['mag_g'])
    MAGERR_G = np.array(table['magerr_g'])
    MAG_R = np.array(table['mag_r'])
    MAGERR_R = np.array(table['magerr_r'])

    c = SkyCoord(
        ra=RA * u.degree,
        dec=DEC * u.degree,
        frame='icrs'
    )
    L = c.galactic.l.degree
    B = c.galactic.b.degree

    MAG_G -= AG_AV * get_av(L, B, ngp, sgp)
    MAG_R -= AR_AV * get_av(L, B, ngp, sgp)

    cond = (MAG_G < mmax) & (MAG_G > mmin) & (
        MAG_G-MAG_R > cmin) & (MAG_G-MAG_R < cmax)

    RA = RA[cond]
    DEC = DEC[cond]
    MAG_G = MAG_G[cond]
    MAG_R = MAG_R[cond]
    MAGERR_G = MAGERR_G[cond]
    MAGERR_R = MAGERR_R[cond]

    col1 = fits.Column(name='RA', format='D', array=RA)
    col2 = fits.Column(name='DEC', format='D', array=DEC)
    col3 = fits.Column(name='MAG_G', format='E', array=MAG_G)
    col4 = fits.Column(name='MAG_R', format='E', array=MAG_R)
    col5 = fits.Column(name='MAGERR_G', format='E', array=MAGERR_G)
    col6 = fits.Column(name='MAGERR_R', format='E', array=MAGERR_R)

    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    des_cat_filepath = Path(results_path, outfile)
    tbhdu.writeto(des_cat_filepath, overwrite=True)

    return RA, DEC, MAG_G, MAGERR_G, MAG_R, MAGERR_R


def dist_ang(ra1, dec1, ra_ref, dec_ref):
    """Calculated the angular distance between (ra1, dec1) and (ra_ref, dec_ref)
    ra-dec in degrees
    ra1-dec1 can be arrays
    ra_ref-dec_ref are scalars
    output is in degrees
    Parameters
    ----------
    ra1, dec1 : lists or arrays
        Positions in the sky
    ra_ref, dec_ref : floats
        Reference position in the sky

    Returns
    -------
    list
        a list of floats with the angular distance in degrees of the
        first couple of inputs to the position of reference.

    """
    costheta = np.sin(np.radians(dec_ref)) * np.sin(np.radians(dec1)) + np.cos(
        np.radians(dec_ref)
    ) * np.cos(np.radians(dec1)) * np.cos(np.radians(ra1 - ra_ref))
    dist_ang = np.arccos(costheta)
    return np.rad2deg(dist_ang)


def download_iso(version, phot_system, Z, age, av_ext, out_file):
    """Submit a request of an isochrone from PADOVA schema, download the file
    requested and uses as an input file to the stellar population of clusters.

    Parameters
    ----------
    version : str
        Version of PADOVA/PARSEC isochrones ('3.4', '3.5' or '3.6')
    phot_system : str
        Photometric system of isochrones. For DES uses 'decam'.
    Z : float
        Amount of metals in mass. Z_sun = 0.019.
    age : float
        Age of isochrone in years (not log age).
    av_ext : float
        Extinction in V band. Usually 0.000.
    out_file : str
        Name of the output file, which is written as 
        output.
    """
    main_pars = "track_parsec=parsec_CAF09_v1.2S&track_colibri=parsec_CAF09_v1.2S_S_LMC_08_web&track_postagb=" + "no&n_inTPC=10&eta_reimers=0.2&kind_interp=1&kind_postagb=-1&kind_tpagb=-1&kind_pulsecycle=" + \
        "0&kind_postagb=-1&kind_mag=2&kind_dust=0&extinction_coeff=constant&extinction_curve=" + \
        "cardelli&kind_LPV=1&dust_sourceM=dpmod60alox40&dust_sourceC=AMCSIC15&imf_file=tab_imf/" + \
        "imf_kroupa_orig.dat&output_kind=0&output_evstage=1&output_gzip=0"
    webserver = "http://stev.oapd.inaf.it"
    if phot_system == 'des':
        phot_system = 'decam'
    try:
        os.system(("wget -o lixo -Otmp --post-data='submit_form=Submit&cmd_version={}&photsys_file=tab_mag_odfnew/tab_mag_{}.dat&photsys_version=YBC&output_kind=0&output_evstage=1&isoc_isagelog=0&isoc_agelow={:.2e}&isoc_ageupp={:.2e}&isod_dage=0&isoc_ismetlog=0&isoc_zlow={:.3f}&isoc_zupp={:.3f}&isod_dz=0&extinction_av={:.3f}&{}' {}/cgi-bin/cmd_{}".format(
            version, phot_system, age, age, Z, Z, av_ext, main_pars, webserver, version)).replace('e+', 'e'))
    except:
        print("No communication with {}".format(webserver))
    with open('tmp') as f:
        aaa = f.readlines()
        for i, j in enumerate(aaa):
            if '/output' in j:
                out_file_tmp = 'output' + \
                    j.split("/output", )[1][0:12] + '.dat'
    os.system("wget -o lixo -O{} {}/tmp/{}".format(out_file_tmp,
              webserver, out_file_tmp))
    os.system("mv {} {}".format(out_file_tmp, out_file))


def get_av(gal_l, gal_b, ngp, sgp):
    """Return extinction (A) in V band based on l, b position. Rv is defined as
    a constant and equal to 3.1.

    return the extinction in the V band since the catalog used for detection is
    extinction free. In the future, may be read the magnitude already corrected
    by extinction in the VAC or catalog.

    Parameters
    ----------
    gal_l : list
        Galactic longitude of the objects (degrees)
    gal_b : list
        Galactic latitude of the objects (degrees)
    ngp : array or map
        Northern Galactic reddening map.
    sgp : array or map
        Southern Galactic reddening map.
    Returns
    -------
    av : list
        a list of Galactic extinction in the V band to each position
    """
    lt = np.radians(gal_l)
    bt = np.radians(gal_b)

    n = np.array(bt > 0).astype(int)
    n[n == 0] = -1
    x = 2048 * ((1 - (n * np.sin(bt))) ** (0.5)) * np.cos(lt) + 2047.5
    y = -2048 * n * ((1 - (n * np.sin(bt))) ** (0.5)) * np.sin(lt) + 2047.5

    x, y = np.round(x, 0).astype(int), np.round(y, 0).astype(int)
    av = np.empty(len(gal_l))

    for idx, num in enumerate(bt):
        if num > 0:
            av[idx] = 3.1 * ngp[y[idx], x[idx]]
        else:
            av[idx] = 3.1 * sgp[y[idx], x[idx]]
    """
    for i in range(len(bt)):
        if bt[i] > 0:
            av[i] = 3.1 * ngp[y[i], x[i]]
        else:
            av[i] = 3.1 * sgp[y[i], x[i]]
    """
    return av


def make_footprint(
    ra_min, ra_max, dec_min, dec_max, nside_ftp, output_path=Path("results")
):
    """Creates a partial HealPix map based on the area selected,
    with nside = nside_ftp

    Parameters
    ----------
    ra_min, ra_max, dec_min, dec_max : float (global)
        Limits in ra and dec (degrees)
    nside_ftp : int (global)
        Nside of the footprint map.

    Returns
    -------
    list
        a list of ipixels in the area selected (inclusive=False)
    """

    vertices = hp.ang2vec(
        [ra_min, ra_max, ra_max, ra_min],
        [dec_min, dec_min, dec_max, dec_max],
        lonlat=True,
    )

    hp_sample = hp.query_polygon(
        nside_ftp, vertices, inclusive=False, fact=64, nest=True, buff=None
    )

    filename = "ftp_4096_nest.fits"
    filepath = Path(output_path, filename)

    # m = np.bincount(hp_sample, minlength=hp.nside2npix(nside_ftp))

    # hp.mollview(m, nest=True, flip='astro')
    SIGNAL = np.ones(len(hp_sample))

    col0 = fits.Column(name="HP_PIXEL_NEST_4096", format="J", array=hp_sample)
    col1 = fits.Column(name="SIGNAL", format="E", array=SIGNAL)

    cols = fits.ColDefs([col0, col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(filepath, overwrite=True)

    return hp_sample


def d_star_real_cat(hpx_ftp, length, nside3, nside_ftp):
    """This function distributes the set of stars with lenght
    in a catalog based on the set of hpx_ftp (healpix footprint).

    Parameters
    ----------
    hpx_ftp : list
        The set of ipixels in the footprint
    length : int
        Total amount of stars in the real catalog
    nside3: int
        Nside HealPix to granullarity of the stellar positions.
    nside_ftp: int
        Nside of the footprint map.

    Returns
    -------
    ra_mw_stars, dec_mw_stars : lists
        The position of the stars in the catalog (degrees)
    """
    f2 = nside3 / nside_ftp
    A = np.repeat(hpx_ftp, f2**2)
    a = np.random.choice(np.arange(f2**2), len(hpx_ftp), replace=True)
    set_pixels_nside3 = A * (f2**2) + a
    hpx_star = np.random.choice(set_pixels_nside3, length, replace=False)
    np.random.shuffle(hpx_star)
    ra_mw_stars, dec_mw_stars = hp.pix2ang(
        nside3, hpx_star, nest=True, lonlat=True)
    return ra_mw_stars, dec_mw_stars


def IMF_(author):
    """Defines dictionary for Kroupa and Salpeter initial mass functions.

    The code below simulates stars using a Kroupa or Salpeter IMF,
    and an exponential radius for the 2D distribution of stars.

    Parameters
    ----------
    author : str
        The name of the initial mass function (IMF)

    Returns
    -------
    dictionary
        a dict with the alpha values and mass breaks
    """
    if author == "Kroupa":
        return {"IMF_alpha_1": -1.3, "IMF_alpha_2": -2.3, "IMF_mass_break": 0.5}
    if author == "Salpeter":
        return {"IMF_alpha_1": -2.3, "IMF_alpha_2": -2.3, "IMF_mass_break": 0.5}

    # TODO: O que acontece se usuario passar um valor diferente? Tratar essa situação para informar o usuario.


def apply_err(mag, mag_table, err_table):
    """This function returns magnitude errors for the 'mag' variable
    based on mag_table and err_table.

    Parameters
    ----------
    mag : list
        The list of magnitudes to be calculated
    mag_table : list
        List of magnitudes
    err_table : List
        List of magnitude errors (1-sigma) respective to mag_table

    Returns
    -------
    list
        a list of magnitude errors following Normal distribution with
        1-sigma error as informed
    """
    err_interp = np.interp(mag, mag_table, err_table)
    return np.abs(err_interp * np.random.randn(len(err_interp)))


def faker_bin(total_bin, IMF_author, file_in, dist):
    """Calculates the fraction of binaries in the simulated clusters.

    Parameters
    ----------
    total_bin : float
        The amount of binaries. Definition: N is the total amount of
        stars (take care to count a system of a binary as two stars),
        and B is the amount of stars in binary systems.
        so bin_frac = B / N
    IMF_author : str
        Name of the IMF (see function above)
    file_in : str
        The name of the file with star's masses and magnitudes
    dist : float
        Distance cluster-observer in parsecs

    Returns
    -------
    binaries[:,0]
        a list of magnitudes of the binaries in the first band
    binaries[:,1]
        a list of magnitudes of the binaries in the second band
    """

    mass, mag1, mag2 = np.loadtxt(file_in, usecols=(3, 29, 30), unpack=True)

    # bin in mass (solar masses)
    binmass = 5.0e-4

    mag1 += 5 * np.log10(dist) - 5
    mag2 += 5 * np.log10(dist) - 5

    IMF = IMF_(IMF_author)

    # amostra is an array with the amount of stars in each bin of mass. ex.: [2,3,4,1,2]
    massmin = np.min(mass)
    massmax = np.max(mass)
    bins_mass = int((massmax - massmin) / binmass)
    amostra = np.zeros(bins_mass)

    for i in range(bins_mass):
        if (i * binmass) + massmin <= IMF["IMF_mass_break"]:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_1"]))
        else:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_2"]))
    # Soma is the total amount of stars (float), the sum of amostra
    soma = np.sum(amostra)
    # Now normalizing the array amostra
    # for idx, num in enumerate(amostra):
    amostra = np.multiply(amostra, total_bin / soma)

    massa_calculada = np.zeros(int(total_bin))

    count = 0

    for j in range(bins_mass):  # todos os intervalos primarios de massa
        for k in range(
            int(amostra[j])
        ):  # amostra() eh a amostra de estrelas dentro do intervalo de massa
            massa_calculada[count] = (
                massmin + (j * binmass) + (k * binmass / amostra[j])
            )
            # massa calculada eh a massa de cada estrela
            count += 1

    # mag1 mag1err unc1 mag2 mag2err unc2
    binaries = np.zeros((total_bin, 2))

    for i in range(total_bin):
        for k in range(len(mass) - 1):  # abre as linhas do arquivo em massa
            # se a massa estiver no intervalo das linhas
            if (mass[k] < massa_calculada[i]) & (mass[k + 1] > massa_calculada[i]):
                # vai abrir tantas vezes quantas forem as estrelas representadas
                intervalo = (massa_calculada[i] - mass[k]) / (
                    mass[k + 1] - mass[k]
                )  # intervalo entre zero e um
                binaries[i, 0] = mag1[k] - (mag1[k] - mag1[k + 1]) * intervalo
                binaries[i, 1] = mag2[k] - (mag2[k] - mag2[k + 1]) * intervalo
    return binaries[:, 0], binaries[:, 1]


def unc(mag, mag_table, err_table):
    """Interpolates the uncertainty in magnitude for a specific magnitude
    using magnitude and error from table.

    Parameters
    ----------
    mag : float or list
        The magnitude to be interpolated
    mag_table : list
        List of magnitudes in table
    err_table : list
        List of magnitude errors in table

    Returns
    -------
    err_interp : float or list
        Magnitudes interpolated
    """
    err_interp = np.interp(mag, mag_table, err_table)
    return err_interp


def faker(
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
    cmin,
    cmax,
    mmin,
    mmax,
    mag1_,
    err1_,
    err2_,
    file_iso,
    output_path
):
    """Creates an array with positions, magnitudes, magnitude errors and magnitude
    uncertainties for the simulated stars in two bands.

    The stars belong to a simple
    stellar population and they are spatially distributed following an exponential profile.
    The code firstly simulates the stars in the CMDs and finally simulates only the
    companions of the binaries (it does not matter how massive the companions are) to
    join to the number of points in the CMD.
    Bear in mind these quantities (definitions):
    N_stars_cmd = number of stars seen in the CMD. The binaries are seen as a single star.
    N_stars_single = amount of stars that are single stars.
    N_stars_bin = amount of stars that are binaries in the CMD. For each of these kind of
    stars, a companion should be calculated later.

    Parameters
    ----------
    N_stars_cmd : int
        Points in simulated cmd given the limiting magnitude. Some of this stars are single,
        some of the are binaries. This amount obeys the following relation:
        N_stars_cmd = N_stars_single + N_stars_bin, where N_stars_single are the single stars
        in the cmd and N_stars_bin are the points in CMD that are binaries. A single star in
        each system are accounted for. In this case, the total amount of stars simulated is
        N_stars_single + 2 * N_stars_bin
    frac_bin : float (0-1)
        Fraction of binaries. This is the total amount of stars in the CMD that belongs to a
        binary system (= 2 * N_stars_bin / total amount of stars).
    IMF_author : str
        Name of the IMF (see function above)
    x0 : float (degrees)
        RA position of the center of cluster
    y0 : float(degrees)
        DEC position of the center of cluster
    rexp : float (degrees)
        Exponential radii of the cluster following the exponential law of density:
        N = A * exp(-r/rexp)
    ell_ : float
        Ellipticity of the cluster (ell_=sqrt((a^2-b^2)/(a^2)))
    pa : float
        Positional angle (from North to East), in degrees
    dist : float
        Distance to the cluster in parsecs
    hpx : int
        Pixel where the cluster resides (nested)
    cmin : float
        Minimum color.
    cmax : float
        Maximum color.
    mmin : float
        Minimum magnitude.
    mmax : float
        Maximum magnitude.
    mag1_ : list
        List of magnitudes acoording error.
    err1_ : list
        List of errors in bluer magnitude.
    err2_ : list
        List of errors in redder magnitude.
    file_iso : str
        Name of file with data from isochrone.
    """

    # Cria o diretório de output se não existir
    os.system('mkdir -p ' + output_path)

    mass, mag1, mag2 = np.loadtxt(file_iso, usecols=(3, 26, 27), unpack=True)

    # bin in mass (solar masses)
    binmass = 5.0e-4

    mag1 += 5 * np.log10(dist) - 5
    mag2 += 5 * np.log10(dist) - 5

    # Warning: cut in mass to avoid faint stars with high errors showing up in the
    # bright part of magnitude. The mass is not the total mass of the cluster,
    # only a lower limit for the total mass.
    cond = mag1 <= mmax + 0.5
    mass, mag1, mag2 = mass[cond], mag1[cond], mag2[cond]

    # amostra is an array with the amount of stars in each bin of mass. ex.: [2,3,4,1,2]
    massmin = np.min(mass)
    massmax = np.max(mass)
    bins_mass = int((massmax - massmin) / binmass)
    amostra = np.zeros(bins_mass)

    IMF = IMF_(IMF_author)

    for i in range(bins_mass):
        if (i * binmass) + massmin <= IMF["IMF_mass_break"]:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_1"]))
        else:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_2"]))
    # Soma is the total amount of stars (float), the sum of amostra
    soma = np.sum(amostra)
    # Now normalizing the array amostra
    for i in range(len(amostra)):
        amostra[i] = N_stars_cmd * amostra[i] / soma

    massa_calculada = np.zeros(int(N_stars_cmd))

    count = 0

    for j in range(bins_mass):  # todos os intervalos primarios de massa
        for k in range(
            int(amostra[j])
        ):  # amostra() eh a amostra de estrelas dentro do intervalo de massa
            massa_calculada[count] = (
                massmin + (j * binmass) + (k * binmass / amostra[j])
            )
            # massa calculada eh a massa de cada estrela
            count += 1

    # 0-RA, 1-DEC, 2-mag1, 3-mag1err, 4-unc1, 5-mag2, 6-mag2err, 7-unc2
    star = np.zeros((N_stars_cmd, 8))

    for i in range(N_stars_cmd):
        for k in range(len(mass) - 1):  # abre as linhas do arquivo em massa
            # se a massa estiver no intervalo das linhas
            if (mass[k] < massa_calculada[i]) & (mass[k + 1] > massa_calculada[i]):
                # vai abrir tantas vezes quantas forem as estrelas representadas
                intervalo = (massa_calculada[i] - mass[k]) / (
                    mass[k + 1] - mass[k]
                )  # intervalo entre zero e um
                star[i, 2] = mag1[k] - (mag1[k] - mag1[k + 1]) * intervalo
                star[i, 5] = mag2[k] - (mag2[k] - mag2[k + 1]) * intervalo

    # apply binarity
    # definition of binarity: fb = N_stars_in_binaries / N_total
    N_stars_bin = int(N_stars_cmd / ((2.0 / frac_bin) - 1))
    mag1_bin, mag2_bin = faker_bin(N_stars_bin, "Kroupa", file_iso, dist)

    j = np.random.randint(N_stars_cmd, size=N_stars_bin)
    k = np.random.randint(N_stars_bin, size=N_stars_bin)

    for j, k in zip(j, k):
        star[j, 2] = -2.5 * np.log10(
            10.0 ** (-0.4 * star[j, 2]) + 10.0 ** (-0.4 * mag1_bin[k])
        )
        star[j, 5] = -2.5 * np.log10(
            10.0 ** (-0.4 * star[j, 5]) + 10.0 ** (-0.4 * mag2_bin[k])
        )

    star[:, 3] = apply_err(star[:, 2], mag1_, err1_)
    star[:, 6] = apply_err(star[:, 5], mag1_, err2_)

    star[:, 4] = unc(star[:, 2], mag1_, err1_)
    star[:, 7] = unc(star[:, 5], mag1_, err2_)

    mag_ref_comp = 22.5
    comp_mag_ref = 1.0
    comp_mag_max = 0.10
    dy_dx = (comp_mag_max - comp_mag_ref) / (mmax - mag_ref_comp)
    p_values = np.zeros(len(star[:, 0]))
    cond = star[:, 2] + star[:, 3] > mag_ref_comp
    p_values[cond] = np.abs(
        (comp_mag_ref - dy_dx * mag_ref_comp)
        + dy_dx * (star[:, 2][cond] + star[:, 3][cond])
    )
    p_values[star[:, 2] > mmax] = 1.0e-9
    p_values[star[:, 2] < mag_ref_comp] = 1.0

    star_comp = np.random.choice(
        len(star[:, 0]), N_stars_cmd, replace=False, p=p_values / np.sum(p_values)
    )

    rexp_deg = (180 / np.pi) * np.arctan(rexp / dist)

    r = exp_prof(rexp_deg, N_stars_cmd)

    phi = 2 * np.pi * np.random.rand(N_stars_cmd)

    X = r * np.sin(phi)
    Y = r * np.cos(phi)
    y_ell = Y
    x_ell = np.multiply(X, (1.0 - ell_))
    r_ell = np.sqrt(x_ell**2 + y_ell**2)
    phi_ell = np.arctan(-x_ell / y_ell)
    phi_ell[x_ell < 0.0] += np.pi
    phi_ell += np.deg2rad(pa)
    star[:, 0] = x0 + (r_ell * np.sin(phi_ell)) / np.cos(np.deg2rad(y0))
    star[:, 1] = y0 + r_ell * np.cos(phi_ell)

    filename = "%s_clus.dat" % str(hpx)
    filepath = Path(output_path, filename)

    with open(filepath, "w") as out_file:
        for ii in star_comp:  # range(len(star[:,0])):
            cor = star[ii, 2] + star[ii, 3] - (star[ii, 5] + star[ii, 6])
            mmag = star[ii, 2] + star[ii, 3]
            if (mmag < mmax) & (mmag > mmin) & (cor >= cmin) & (cor <= cmax):
                print(
                    star[ii, 0],
                    star[ii, 1],
                    star[ii, 2] + star[ii, 3],
                    star[ii, 4],
                    star[ii, 5] + star[ii, 6],
                    star[ii, 7],
                    star[ii, 3],
                    star[ii, 4],
                    file=out_file,
                )


def join_cat(
    ra_min,
    ra_max,
    dec_min,
    dec_max,
    hp_sample_un,
    survey,
    RA,
    DEC,
    MAG_G,
    MAG_R,
    MAGERR_G,
    MAGERR_R,
    nside_ini,
    mmax,
    mmin,
    cmin,
    cmax,
    input_path=Path("results/fake_clus"),
    output_path=Path("results"),
):
    """Join the catalog of real stars with random position
    and all the small catalogs of simulated clusters into a single one.

    There is a cut in the coord limits implemented below.
    Only global parameters are needed (see its description above)
    and the code is intended to write a single fits file.

    Parameters
    ----------
    ra_min : float
        Minimum right ascention in degrees
    ra_max : float
        Maximum right ascention in degrees
    dec_min : float
        Minimum declination in degrees.
    dec_max : float
        Maximum declination in degrees.
    hp_sample_un : list
        List of footprint healpix
    survey : str
        THe name of survey.
    RA : array-like
        Right ascention of the stars, in degrees.
    DEC : array-like
        Declination of stars in degrees.
    MAG_G : array-like
        Bluer magnitude of stars.
    MAG_R : array-like
        Redder magnitude of stars.
    MAGERR_G : array-like
        Error of stellar measurements for the first band.
    MAGERR_R : array-like
        Error of stellar measurements for the second band.
    nside_ini : float
        Nside of hp_sample_un.
    mmax : float
        Maximum magnitude of stars.
    mmin : float
        Minimum magnitude of stars.
    cmin : float
        Minimum color of stars.
    cmax : float
        Maximum color of stars.
    """

    GC = np.zeros(len(RA), dtype=int)

    for j in range(len(hp_sample_un)):
        try:
            # input_path = Diretório onde se encontram os arquivos _clus.
            filepath = Path(input_path, "%s_clus.dat" % hp_sample_un[j])
            (
                RA_clus,
                DEC_clus,
                MAG1_clus,
                MAGERR1_clus,
                MAG2_clus,
                MAGERR2_clus,
            ) = np.loadtxt(
                filepath,
                usecols=(0, 1, 2, 3, 4, 5),
                unpack=True,
            )

            pr_limit = (
                (RA_clus >= ra_min)
                & (RA_clus <= ra_max)
                & (DEC_clus >= dec_min)
                & (DEC_clus <= dec_max)
                & (MAG1_clus <= mmax)
                & (MAG1_clus >= mmin)
                & (MAG1_clus - MAG2_clus >= cmin)
                & (MAG1_clus - MAG2_clus <= cmax)
            )

            RA_clus, DEC_clus, MAG1_clus, MAG2_clus, MAGERR1_clus, MAGERR2_clus = (
                RA_clus[pr_limit],
                DEC_clus[pr_limit],
                MAG1_clus[pr_limit],
                MAG2_clus[pr_limit],
                MAGERR1_clus[pr_limit],
                MAGERR2_clus[pr_limit],
            )

            GC_clus = np.ones(len(RA_clus), dtype=int)
            GC = np.concatenate((GC, GC_clus), axis=0)
            RA = np.concatenate((RA, RA_clus), axis=0)
            DEC = np.concatenate((DEC, DEC_clus), axis=0)
            MAG_G = np.concatenate((MAG_G, MAG1_clus), axis=0)
            MAG_R = np.concatenate((MAG_R, MAG2_clus), axis=0)
            MAGERR_G = np.concatenate((MAGERR_G, MAGERR1_clus), axis=0)
            MAGERR_R = np.concatenate((MAGERR_R, MAGERR2_clus), axis=0)
        except:
            print("zero stars in ", hp_sample_un[j])

    filepath = Path(output_path, "%s_mockcat_for_detection.fits" % survey)

    HPX64 = hp.ang2pix(nside_ini, RA, DEC, nest=True, lonlat=True)
    col0 = fits.Column(name="GC", format="I", array=GC)
    col1 = fits.Column(name="ra", format="D", array=RA)
    col2 = fits.Column(name="dec", format="D", array=DEC)
    col3 = fits.Column(name="mag_g_with_err", format="E", array=MAG_G)
    col4 = fits.Column(name="mag_r_with_err", format="E", array=MAG_R)
    col5 = fits.Column(name="magerr_g", format="E", array=MAGERR_G)
    col6 = fits.Column(name="magerr_r", format="E", array=MAGERR_R)
    col7 = fits.Column(name="HPX64", format="K", array=HPX64)
    cols = fits.ColDefs([col0, col1, col2, col3, col4, col5, col6, col7])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(filepath, overwrite=True)

    return filepath


def snr_estimate(
    RA__,
    DEC__,
    G__,
    GR__,
    PIX_sim,
    nside1,
    mM_,
    inner_circle,
    rin_annulus,
    rout_annulus,
):
    """
    Estimate the SNR (Signal to Noise Ratio) of the simulated cluster in density-number.


    Parameters
    ----------
    RA__, DEC__ : lists
        The coordinates of the stars in the catalog
    G__, GR__ : lists
        Apparent magnitude in g band and g-r color of the stars.
    PIX_sim : int
        The pixel where the simulated cluster resided (nside=nside1)
        Nest=True.
    nside1 : int
        Nside of the pixelization of the sky reagarding the distribution of simulated clusters.
    mM_ : float
        Modulus distance of the simulated cluster
    inner_circle : float
        Radius of the inner circle (in deg) where the signal is estimated
    rin_annulus : float
        Inner radius where the noise is estimated (in deg)
    rout_annulus : float
        Outer radius where the noise is estimated (in deg)

    Returns
    -------
    float
        The SNR estimated following the SNR = N_inner / sqrt(N_outter_circle)
        where N_inner is the amount of stars in the inner_circle
        and N_outter_circle is the star counts in the outter circle
        normalized by the area of the inner_circle.
    """
    ra_center, dec_center = hp.pix2ang(nside1, PIX_sim, nest=True, lonlat=True)
    # loading data from isochronal mask
    gr_mask, g_mask, kind_mask = np.loadtxt(
        "gr_g_model_D0.asc", usecols=(0, 1, 2), unpack=True
    )
    g_mask += mM_

    # TODO: Verificar esses arquivos hardcoded. se deveriam ser parametros
    # OU estar em um path fixo dentro da lib
    # Estes arquivos informam o erro em magnitude. Variam de survey oara survey
    # e de release a release. Talvez fazer pastas com o nome do survey e colocar
    # eles apenas como mag1_err e mag2_err fosse melhor. ou informar no set de
    # parametros.
    _g, _gerr = np.loadtxt("des_y6_g_gerr.asc", usecols=(0, 1), unpack=True)
    _r, _rerr = np.loadtxt("des_y6_r_rerr.asc", usecols=(0, 1), unpack=True)

    for i in range(len(gr_mask)):
        err_ = np.sqrt(
            np.interp(g_mask[i], _g, _gerr) ** 2.0
            + np.interp(g_mask[i] - gr_mask[i], _r, _rerr) ** 2.0
        )
        if err_ <= 1.00:
            if kind_mask[i] == 0.0:
                gr_mask[i] -= err_
            elif kind_mask[i] == 1.0:
                gr_mask[i] += err_

    points = np.column_stack([GR__, G__])
    verts = np.array([gr_mask, g_mask]).T
    path = mpath.Path(verts)
    t = path.contains_points(points)
    RA_ = list(compress(RA__, t))
    DEC_ = list(compress(DEC__, t))

    area_inner_circle = np.pi * (inner_circle**2)
    area_annulus = np.pi * ((rout_annulus**2) - (rin_annulus**2))
    r_star = np.sqrt(
        ((RA_ - ra_center) * np.cos(np.deg2rad(dec_center))) ** 2
        + (DEC_ - dec_center) ** 2
    )
    N_bg_equal_area = float(
        len(r_star[(r_star > rin_annulus) & (r_star < rout_annulus)])
        * area_inner_circle
        / area_annulus
    )

    return len(r_star[r_star < inner_circle]) / np.sqrt(N_bg_equal_area)


def write_sim_clus_features(
    mockcat, hp_sample_un, nside_ini, mM, output_path=Path("results")
):
    """
    Write a few features of the clusters in a file called 'N_stars,dat'.
    The columns of the file are:
    - the ipix of the cluster that serves as an ID (hp_sample_un[j]),
    - star counts (len(RA[cond_clus])),
    - absolute magnitude in V band (M_abs_V), and
    - signal-to-noise ratio (SNR).
    Only global parameters are needed (see its description in the first cells).
    The file must be written after the simulation because the absolute
    magnitude of the cluster and the number of stars are estimated after
    the simulation. The absolute magnitude depends strongly on the brightest
    stars and the star counts may vary in a few counts for instance
    for two clusters with the same mass (stars are a numerical realization
    within an IMF).
    """

    # hdu = fits.open(survey + "_mockcat_for_detection.fits", memmap=True)
    hdu = fits.open(mockcat, memmap=True)
    GC = hdu[1].data.field("GC")
    RA = hdu[1].data.field("ra")
    DEC = hdu[1].data.field("dec")
    MAG_G = hdu[1].data.field("mag_g_with_err")
    MAG_R = hdu[1].data.field("mag_r_with_err")
    HPX64 = hdu[1].data.field("HPX64")

    filepath = Path(output_path, "n_stars.dat")

    with open(filepath, "w") as out_file:
        for j in range(len(hp_sample_un)):
            # try:
            cond = HPX64 == hp_sample_un[j]
            RA__, DEC__, MAGG__, MAGR__ = RA[cond], DEC[cond], MAG_G[cond], MAG_R[cond]
            # plt.scatter(RA__, DEC__)
            # plt.show()
            SNR = snr_estimate(
                RA__,
                DEC__,
                MAGG__,
                MAGG__ - MAGR__,
                hp_sample_un[j],
                nside_ini,
                mM[j],
                2.0 / 60,
                10.0 / 60,
                25.0 / 60,
            )

            cond_clus = (cond) & (GC == 1)
            # TODO: Variaveis declaradas e não usadas.
            MAGG_clus, MAGR_clus = (
                MAG_G[cond_clus],
                MAG_R[cond_clus],
            )
            flux_g = 10 ** (-0.4 * MAGG_clus)
            flux_r = 10 ** (-0.4 * MAGR_clus)
            M_abs_g = -2.5 * np.log10(np.sum(flux_g)) - mM[j]
            M_abs_r = -2.5 * np.log10(np.sum(flux_r)) - mM[j]
            M_abs_V = (
                M_abs_g - 0.58 * (M_abs_g - M_abs_r) - 0.01
            )  # in V band following Jester 2005

            print(
                "{:d} {:d} {:.2f} {:.2f}".format(
                    hp_sample_un[j], len(RA[cond_clus]), M_abs_V, SNR
                ),
                file=out_file,
            )
    return filepath


def SplitFtpHPX(
    file_in, out_dir, nside_in=4096, nest_in=True, nside_out=64, nest_out=True
):
    # TODO: Usar pathlib com a opção que já verifica se o diretório existe
    try:
        os.mkdir(out_dir)
    except:
        print("Folder already exists or is an invalid name.")

    data = getdata(file_in)
    HPX4096 = data["HP_PIXEL_NEST_4096"]
    SIGNAL = data["SIGNAL"]

    ra, dec = hp.pix2ang(nside_in, HPX4096, nest=nest_in, lonlat=True)

    HPX64 = hp.ang2pix(nside_out, ra, dec, nest=nest_out, lonlat=True)

    HPX_un = set(HPX64)

    for i in HPX_un:

        cond = HPX64 == i

        col0 = fits.Column(name="HP_PIXEL_NEST_4096",
                           format="K", array=HPX4096[cond])
        col1 = fits.Column(name="SIGNAL", format="E", array=SIGNAL[cond])
        cols = fits.ColDefs([col0, col1])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(out_dir + "/" + str(int(i)) + ".fits", overwrite=True)


def radec2GCdist(ra, dec, dist_kpc):
    """
    Return Galactocentric distance from ra, dec, D_sun_kpc.

    Parameters
    ----------
    ra, dec : float or list
        Coordinates of the objects (in deg)
    dist_kpc : float or list
        Distance in kpc of the objects

    Returns
    -------
    float of list
        the Galactocentric distance to the object[s]
    """
    c1 = coord.SkyCoord(
        ra=ra * u.degree, dec=dec * u.degree, distance=dist_kpc * u.kpc, frame="icrs"
    )
    x, y, z = (
        c1.transform_to(coord.Galactocentric).x.value,
        c1.transform_to(coord.Galactocentric).y.value,
        c1.transform_to(coord.Galactocentric).z.value,
    )

    return np.sqrt(x * x + y * y + z * z)


def remove_close_stars(input_cat, output_cat, nside_ini, PSF_factor, PSF_size):
    # TODO: this function is really slow. Improve that function to remove
    # star closer than an specific distance in parallel.
    """This function removes the stars closer than PSF_factor * PSF_size
    This is an observational bias of the DES since the photometric pipeline
    is set to join regions closer than an specific distance.
    In this sense, if many objects are crowded in a small area,
    the pipeline joins the detections into a single object.

    This function may be improved to run in parallel using many ipixels
    since it takes too long to check and remove stars.

    Parameters
    ----------
    input_cat : str
        The file with the information of the objects
    output_cat : str
        The output file to be written with the stars that survived the test.
    PSF_factor : float
        A factor to multiply the PSF size. The result is the minimal distance
        between two objects to be detected as two objects. If the distance
        is less than PSF_factor * PSF_size, none of the objects survives.

    """
    input_cat = Path(input_cat)
    output_cat = Path(output_cat)

    hdu = fits.open(input_cat, memmap=True)
    GC = hdu[1].data.field("GC")
    ra = hdu[1].data.field("ra")
    dec = hdu[1].data.field("dec")
    mag_g_with_err = hdu[1].data.field("mag_g_with_err")
    mag_r_with_err = hdu[1].data.field("mag_r_with_err")
    magerr_g = hdu[1].data.field("magerr_g")
    magerr_r = hdu[1].data.field("magerr_r")
    HPX64 = hdu[1].data.field("HPX64")
    hdu.close()

    idx = []
    seplim = (PSF_factor * PSF_size / 3600) * u.degree
    for i in range(len(ra)):
        cond = (
            (ra > ra[i] - 0.05)
            & (ra < ra[i] + 0.05)
            & (dec > dec[i] - 0.05)
            & (dec < dec[i] + 0.05)
        )

        c = SkyCoord(ra=ra[cond] * u.degree, dec=dec[cond] * u.degree)
        idx_, sep2d, dist3d = match_coordinates_sky(
            c, c, nthneighbor=2, storekdtree="kdtree_sky"
        )
        cond2 = [sep2d < seplim]
        if len(idx_[cond2]) == 0:
            idx.append(i)
        # dists = dist_ang(ra[cond], dec[cond], ra[i], dec[i])
        # dist = np.sort(list(dists))[1]
        # if dist > (PSF_factor * PSF_size/3600.):
        #    idx.append(i)

    HPX64 = hp.ang2pix(nside_ini, ra, dec, nest=True, lonlat=True)
    col0 = fits.Column(name="GC", format="I",
                       array=np.asarray([GC[i] for i in idx]))
    col1 = fits.Column(name="ra", format="D",
                       array=np.asarray([ra[i] for i in idx]))
    col2 = fits.Column(name="dec", format="D",
                       array=np.asarray([dec[i] for i in idx]))
    col3 = fits.Column(
        name="mag_g_with_err",
        format="E",
        array=np.asarray([mag_g_with_err[i] for i in idx]),
    )
    col4 = fits.Column(
        name="mag_r_with_err",
        format="E",
        array=np.asarray([mag_r_with_err[i] for i in idx]),
    )
    col5 = fits.Column(
        name="magerr_g", format="E", array=np.asarray([magerr_g[i] for i in idx])
    )
    col6 = fits.Column(
        name="magerr_r", format="E", array=np.asarray([magerr_r[i] for i in idx])
    )
    col7 = fits.Column(
        name="HPX64", format="K", array=np.asarray([HPX64[i] for i in idx])
    )
    cols = fits.ColDefs([col0, col1, col2, col3, col4, col5, col6, col7])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(output_cat, overwrite=True)

    return output_cat
