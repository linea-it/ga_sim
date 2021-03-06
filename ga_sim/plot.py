# -*- coding: utf-8 -*-
import astropy.io.fits as fits
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import getdata
from pathlib import Path
from ga_sim.ga_sim import radec2GCdist
mpl.rcParams["legend.numpoints"] = 1


def plot_clusters_clean(ipix_cats, ipix_clean_cats, nside, ra_str, dec_str, half_size_plot=0.01):
    """_summary_

    Parameters
    ----------
    ipix_cats : list
        List of catalogs with all stars.
    ipix_clean_cats : list
        List of catalogs with stars filtered.
    nside : int
        Nside of pixelizations.
    half_size_plot : float, optional
        Size to be seen on plots. Usually twice the angular size of exponential
        profiles of clusters. Units: degrees.
    """
    len_ipix = len(ipix_clean_cats)

    ipix = [int((i.split('/')[-1]).split('.')[0]) for i in ipix_cats]

    ra_cen, dec_cen = hp.pix2ang(nside, ipix, nest=True, lonlat=True)
    half_size_plot = 0.01
    tot_clus = 0
    for i in range(len_ipix):
        data = fits.getdata(ipix_cats[i])
        RA_orig = data[ra_str]
        DEC_orig = data[dec_str]
        if len(RA_orig[(RA_orig < ra_cen[i] + half_size_plot) & (RA_orig > ra_cen[i] - half_size_plot) &
                       (DEC_orig < dec_cen[i] + half_size_plot) & (DEC_orig > dec_cen[i] - half_size_plot)]) > 10.:
            tot_clus += 1
    fig, ax = plt.subplots(int(tot_clus / 4) + 1, 4, figsize=(16, tot_clus))
    j = 0
    for i in range(len(ax[:,0])):
        for k in range(len(ax[0, :])):
            ax[i, k].set_xticks([])
            ax[i, k].set_yticks([])

    for i in range(len_ipix):
        data = fits.getdata(ipix_cats[i])
        RA_orig = data[ra_str]
        DEC_orig = data[dec_str]

        if len(RA_orig[(RA_orig < ra_cen[i] + half_size_plot) & (RA_orig > ra_cen[i] - half_size_plot) &
                       (DEC_orig < dec_cen[i] + half_size_plot) & (DEC_orig > dec_cen[i] - half_size_plot)]) > 10.:
            line = int(j / 4)
            col = int(j % 4)
            data = fits.getdata(ipix_clean_cats[i])
            RA = data[ra_str]
            DEC = data[dec_str]
            ax[line, col].scatter(
                RA_orig, DEC_orig, edgecolor='b', color='None', s=20, label='All stars')
            ax[line, col].set_xlim(
                [ra_cen[i] + half_size_plot, ra_cen[i] - half_size_plot])
            ax[line, col].set_ylim(
                [dec_cen[i] - half_size_plot, dec_cen[i] + half_size_plot])
            ax[line, col].scatter(RA, DEC, color='r', s=2, label='Filtered stars')
            ax[line, col].set_xlim(
                [ra_cen[i] + half_size_plot, ra_cen[i] - half_size_plot])
            ax[line, col].set_ylim(
                [dec_cen[i] - half_size_plot, dec_cen[i] + half_size_plot])
            ax[line, col].set_title('Ipix='+str(ipix[i]), y= 0.9, pad=8) #{x=ra_cen[i], y=dec_cen[i], pad=8)
            ax[line, col].legend(loc=3)
            j += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    
def general_plots(star_clusters_simulated):

    PIX_sim, NSTARS, MAG_ABS_V, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(
        star_clusters_simulated,
        usecols=(0, 1, 2, 6, 7, 8, 9, 10, 11, 12),
        unpack=True,
    )
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.scatter(np.log10(1.7 * R_EXP[MAG_ABS_V < 0.0]), MAG_ABS_V[MAG_ABS_V < 0.0])
    ax1.set_ylabel("M(V)")
    ax1.set_xlabel("log10(h-l radii(pc))")

    ax2.scatter(MASS, MAG_ABS_V)
    ax2.set_xlabel("mass(Msun)")
    plt.show()


def plot_ftp(
    ftp_fits, star_clusters_simulated, mockcat, ra_max, ra_min, dec_min, dec_max
):
    """Plot footprint map to check area."""
    nside = 4096
    npix = hp.nside2npix(nside)

    # data = getdata("ftp_4096_nest.fits")
    data = getdata(ftp_fits)

    pix_ftp = data["HP_PIXEL_NEST_4096"]

    ra_pix_ftp, dec_pix_ftp = hp.pix2ang(nside, pix_ftp, nest=True, lonlat=True)
    map_ftp = np.zeros(hp.nside2npix(nside))
    map_ftp[pix_ftp] = 1

    test = hp.cartview(
        map_ftp,
        nest=True,
        lonra=[np.min(ra_pix_ftp), np.max(ra_pix_ftp)],
        latra=[np.min(dec_pix_ftp), np.max(dec_pix_ftp)],
        hold=True,
        cbar=False,
        title="",
        return_projected_map=True,
    )
    # plt.clf()

    # TODO: Verificar variaveis carregadas e n??o usadas
    PIX_sim, NSTARS, MAG_ABS_V, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(
        star_clusters_simulated,
        usecols=(0, 1, 2, 6, 7, 8, 9, 10, 11, 12),
        unpack=True,
    )
    for i in range(len(RA)):
        hp.projtext(
            RA[i],
            DEC[i],
            str(PIX_sim[i]),
            lonlat=True,
            fontsize=10,
            c="k",
            horizontalalignment="center",
        )
        hp.projscatter(RA[i], DEC[i], lonlat=True, coord="C", s=1.0, color="k", lw=0.1)

    # data = getdata(survey + "_mockcat_for_detection.fits")
    data = getdata(mockcat)
    RA_star, DEC_star = data["RA"], data["DEC"]
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(
        test,
        origin="lower",
        extent=(ra_max, ra_min, dec_min, dec_max),
        interpolation="none",
    )
    axs.scatter(RA, DEC, s=20, c="k", marker="s", label="Simulated clusters")
    axs.scatter(RA_star, DEC_star, s=0.1, c="k", marker="o", label="Simulated stars")
    axs.set_xlim([ra_min, ra_max])
    axs.set_ylim([dec_min, dec_max])
    axs.set_xlabel("RA (deg)")
    axs.set_ylabel("DEC (deg)")
    axs.set_title("Distribution of stars on Footprint Map")
    axs.grid()
    plt.legend()
    plt.show()
    print(len(PIX_sim))


def plots_ang_size(
    star_clusters_simulated,
    clus_path,
    mmin,
    mmax,
    cmin,
    cmax,
    output_plots=Path("results"),
):
    """Plots to analyze the simulated clusters."""

    # TODO: Seria interessante dividir essa fun????o em duas
    # Uma para os plots de _clus.dat
    # Outra para os 4 plots do final.
    # Dessa forma a gera????o dos plots _clus.dat poderia ser paralelizada.

    cmap = mpl.cm.get_cmap("inferno")
    cmap.set_under("dimgray")
    cmap.set_bad("black")

    # TODO: Variaveis Instanciadas e n??o usadas
    (
        hp_sample_un,
        NSTARS,
        MAG_ABS_V,
        RA_pix,
        DEC_pix,
        r_exp,
        ell,
        pa,
        mass,
        dist,
    ) = np.loadtxt(
        star_clusters_simulated,
        usecols=(0, 1, 2, 6, 7, 8, 9, 10, 11, 12),
        unpack=True,
    )

    # Cria um diret??rio para os plots caso n??o exista
    output_plots = Path(output_plots)
    output_plots.mkdir(parents=True, exist_ok=True)

    for i in hp_sample_un:

        clus_filepath = Path(clus_path, "%s_clus.dat" % int(i))
        plot_filepath = Path(output_plots, "%s_cmd.png" % int(i))

        # Evita erro se o arquivo _clus.dat n??o existir.
        # Pode acontecer se o teste estiver usando uma quantidade pequena de dados.
        if not clus_filepath.exists():
            continue

        star = np.loadtxt(clus_filepath)
        """
        plt.scatter(star[:,2]-star[:,4], star[:,2], color='b')
        plt.title('HPX ' + str(int(ii)) + ', N=' + str(len(star[:,2])))
        plt.ylim([mmax, mmin])
        plt.xlim([cmin, cmax])
        plt.xlabel('mag1-mag2')
        plt.ylabel('mag1')
        plt.savefig(str(int(ii)) + '_cmd.png')
        plt.close()
        """
        h1, xedges, yedges, im1 = plt.hist2d(
            star[:, 2] - star[:, 4],
            star[:, 2],
            bins=50,
            range=[[cmin, cmax], [mmin, mmax]],
            # norm=mpl.colors.LogNorm(),
            cmap=cmap,
        )
        plt.clf()
        plt.title("HPX " + str(int(i)) + ", N=" + str(len(star[:, 2])))
        im1 = plt.imshow(
            h1.T,
            interpolation="None",
            origin="lower",
            vmin=0.1,
            vmax=np.max(h1),
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap=cmap,
        )
        plt.ylim([mmax, mmin])
        plt.xlim([cmin, cmax])
        plt.xlabel("mag1-mag2")
        plt.ylabel("mag1")
        plt.colorbar(im1, cmap=cmap, orientation="vertical", label="stars per bin")
        plt.savefig(plot_filepath)
        plt.close()

    ang_size = 60 * np.rad2deg(np.arctan(1.7 * r_exp / dist))
    plt.hist(dist / 1000, bins=50)
    plt.xlabel("Distance (kpc)")
    plt.ylabel("N objects")
    plt.show()
    plt.hist(ang_size, bins=50)
    plt.xlabel("Half-light radii (arcmin)")
    plt.ylabel("N objects")
    plt.show()
    plt.scatter(dist / 1000, ang_size)
    plt.xlabel("Distance (kpc)")
    plt.ylabel("Half-light radii (arcmin)")
    plt.show()

    plt.scatter(mass, NSTARS)
    plt.xlabel("MASS(MSun)")
    plt.ylabel("N stars")
    plt.show()

    plt.scatter(mass, MAG_ABS_V)
    plt.xlabel("MASS(MSun)")
    plt.ylabel("MAG_ABS_V")
    plt.show()


def plots_ref(
    FeH_iso,
    star_clusters_simulated=Path("results/star_clusters_simulated.dat"),
    output_plots=Path("results"),
):
    """Make a few plots about the simulated clusters"""
    # TODO: Talves separar os plots em fun????es diferentes.

    catalogs_path = Path("catalogs")

    # Catalogo objects_in_ref.dat
    obj_ref_filepath = Path(catalogs_path, "objects_in_ref.dat")
    ra_DG, dec_DG, dist_kpc_obj, Mv_obj, rhl_pc_obj, FeH_DG = np.loadtxt(
        obj_ref_filepath, usecols=(0, 1, 4, 8, 10, 11), unpack=True
    )

    # # TODO: Variavel instanciada e n??o utilizada
    # name_obj = np.loadtxt(
    #     "catalogos/objects_in_ref.dat", dtype=str, usecols=(2), unpack=True
    # )

    #  Catalogo Harris_updated.dat
    harris_updated_filepath = Path(catalogs_path, "Harris_updated.dat")
    # 0-Name 1-L 2-B 3-R_gc	4-Fe/H 5-M-M 6-Mv 7-rhl arcmin
    R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_arcmin_GC = np.loadtxt(
        harris_updated_filepath, usecols=(3, 4, 5, 6, 7), unpack=True
    )
    dist_kpc_GC = 10 ** (mM_GC / 5 - 2)

    # Star Clusters Simulated
    # TODO: Variaveis instanciadas e n??o utilizadas
    star_clusters_simulated = Path(star_clusters_simulated)
    PIX_sim, NSTARS, MAG_ABS_V, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(
        star_clusters_simulated,
        usecols=(0, 1, 2, 6, 7, 8, 9, 10, 11, 12),
        unpack=True,
    )
    LOG10_RHL_PC_SIM = np.log10(1.7 * R_EXP)
    rhl_pc_GC = 1000 * dist_kpc_GC * (rhl_arcmin_GC / (57.3 * 60))

    MW_center_distance_DG_kpc = radec2GCdist(ra_DG, dec_DG, dist_kpc_obj)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs[0, 0].hist(
        MAG_ABS_V,
        bins=20,
        range=(-16, 0.0),
        histtype="stepfilled",
        label="Sim",
        color="grey",
        ls="--",
    )
    axs[0, 0].hist(
        Mv_obj, bins=20, range=(-16, 0.0), histtype="step", label="DG", color="r"
    )
    axs[0, 0].hist(
        Mv_GC, bins=20, range=(-16, 0.0), histtype="step", label="GC", color="b"
    )
    axs[0, 0].set_xlabel(r"$M_V$")
    axs[0, 0].set_ylabel("N")
    axs[0, 0].legend(loc=2)

    axs[0, 1].hist(
        LOG10_RHL_PC_SIM,
        bins=20,
        histtype="stepfilled",
        range=(0, 4.0),
        label="Sim",
        color="grey",
        ls="--",
    )
    axs[0, 1].hist(
        np.log10(rhl_pc_obj),
        bins=20,
        histtype="step",
        range=(0, 4.0),
        label="DG",
        color="r",
    )
    axs[0, 1].hist(
        np.log10(rhl_pc_GC),
        bins=20,
        histtype="step",
        range=(0, 4.0),
        label="GC",
        color="b",
    )
    axs[0, 1].set_xlabel("log10(rhl[pc])")
    axs[0, 1].legend(loc=1)

    axs[1, 0].hist(
        DIST / 1000,
        bins=20,
        range=(0, 400.0),
        histtype="stepfilled",
        label="Sim",
        color="grey",
        ls="--",
    )
    axs[1, 0].hist(
        dist_kpc_obj, bins=20, range=(0, 400.0), histtype="step", label="DG", color="r"
    )
    axs[1, 0].hist(
        dist_kpc_GC, bins=20, range=(0, 400.0), histtype="step", label="GC", color="b"
    )
    axs[1, 0].set_xlabel("Distance (kpc)")
    axs[1, 0].legend(loc=1)

    axs[1, 1].hist(
        np.repeat(FeH_iso, len(MAG_ABS_V)),
        bins=20,
        range=(-3, 1.0),
        histtype="stepfilled",
        label="Sim",
        color="grey",
        ls="--",
    )
    axs[1, 1].hist(
        FeH_GC, bins=20, range=(-3, 1.0), histtype="step", label="GC", color="r"
    )
    axs[1, 1].hist(
        FeH_DG, bins=20, range=(-3, 1.0), histtype="step", label="DG", color="b"
    )
    axs[1, 1].set_xlabel("[Fe/H]")
    axs[1, 1].legend(loc=1)

    #  PLOT 1 ----------------
    plt.suptitle(
        "Physical features of 58 Dwarf Gal + 152 GC + "
        + str(len(PIX_sim))
        + " Simulations",
        fontsize=16,
    )
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)

    filepath = Path(output_plots, "_01_real_objects.png")
    plt.savefig(filepath)
    plt.close()

    #  PLOT 2 ----------------
    plt.scatter(
        DIST / 1000,
        np.repeat(FeH_iso, len(DIST)),
        label="Sim",
        color="grey",
        marker="x",
        lw=1.0,
    )
    plt.scatter(MW_center_distance_DG_kpc, FeH_DG, label="DG", color="r")
    plt.scatter(R_MW_GC, FeH_GC, label="GC", color="b")
    plt.xlabel("Distance to the Galactic center (kpc)")
    plt.ylabel("[Fe/H]")
    plt.ylim([-3.5, 0])
    plt.legend()
    plt.grid()
    filepath = Path(output_plots, "_02_feh_rgc.png")
    plt.savefig(filepath)
    plt.close()

    # TODO: Variavel instanciada e n??o utilizada
    rhl = np.logspace(np.log10(1.8), np.log10(1800), 10, endpoint=True)
    m_v = np.linspace(1, -14, 10, endpoint=True)

    #  PLOT 3 ----------------
    plt.scatter(1.7 * R_EXP, MAG_ABS_V, marker="s", color="grey", label="Sim")
    plt.scatter(rhl_pc_obj, Mv_obj, marker="^", color="r", label="DG")
    plt.scatter(rhl_pc_GC, Mv_GC, marker="x", color="b", label="GC")
    plt.plot(
        np.logspace(np.log10(1.8), np.log10(1800), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True),
        color="b",
        ls=":",
    )
    plt.plot(
        np.logspace(np.log10(4.2), np.log10(4200), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True),
        color="b",
        ls=":",
    )
    plt.plot(
        np.logspace(np.log10(11), np.log10(11000), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True),
        color="b",
        ls=":",
    )
    plt.plot(
        np.logspace(np.log10(28), np.log10(28000), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True),
        color="b",
        ls=":",
    )
    plt.text(300, -7.9, r"$\mu_V=27\ mag/arcsec$", rotation=40)
    plt.text(400, -4.2, r"$\mu_V=31\ mag/arcsec$", rotation=40)
    plt.xscale("log")
    plt.xlim([0.4, 4000])
    plt.ylim([1, -14])
    plt.ylabel(r"$M_V$")
    plt.xlabel(r"$r_h\ (pc)$")
    plt.legend(loc=2, frameon=True)

    filepath = Path(output_plots, "_03_mv_rh.png")
    plt.savefig(filepath)
    plt.close()

    plt.show()


def plot_err(
    mockcat=Path("results/des_mockcat_for_detection.fits"), output_plots=Path("results")
):

    """Plot the magnitude and error of the simulated clusters compared to the
    real stars, in log scale.

    """
    # TODO: Path de arquivo hardcoded, verificar se pode ser parametro ou um path fixo
    mockcat = Path(mockcat)

    hdu = fits.open(mockcat, memmap=True)
    GC = hdu[1].data.field("GC")

    # TODO: Variaveis instanciadas mas n??o utilizadas
    ra = hdu[1].data.field("ra")
    dec = hdu[1].data.field("dec")
    mag_g_with_err = hdu[1].data.field("mag_g_with_err")
    mag_r_with_err = hdu[1].data.field("mag_r_with_err")
    magerr_g = hdu[1].data.field("magerr_g")
    magerr_r = hdu[1].data.field("magerr_r")
    hdu.close()

    plt.scatter(mag_g_with_err[GC == 0], magerr_g[GC == 0], label="Field stars", c="k")
    plt.scatter(
        mag_g_with_err[GC == 1],
        magerr_g[GC == 1],
        label="Simulated stars",
        c="r",
        zorder=10,
    )
    plt.yscale("log")
    plt.xlabel("mag_g_with_err")
    plt.ylabel("magerr_g")
    plt.legend()

    filepath = Path(output_plots, "simulated_stars_err.png")
    plt.savefig(filepath)

    plt.show()
