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


def read_real_cat(cat_DG = "catalogs/objects_in_ref.dat", cat_GC = "catalogs/Harris_updated.dat"):

    ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG = np.loadtxt(
        cat_DG, usecols=(0, 1, 4, 8, 10, 11), unpack=True
    )

    name_DG = np.loadtxt(
        cat_DG, dtype=str, usecols=(2), unpack=True
    )

    #  Catalogo Harris_updated.dat
    # 0-Name 1-L 2-B 3-R_gc	4-Fe/H 5-M-M 6-Mv 7-rhl arcmin
    R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_arcmin_GC = np.loadtxt(
        cat_GC, usecols=(3, 4, 5, 6, 7), unpack=True
    )
    
    dist_kpc_GC = 10 ** ((mM_GC / 5) - 2)
    
    rhl_pc_GC = 1000 * dist_kpc_GC * np.tan(rhl_arcmin_GC / (60 * 180 / np.pi))
    
    name_GC = np.loadtxt(
        cat_GC, dtype=str, usecols=(0), unpack=True
    )

    return name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC


def plot_clusters_clean(ipix_cats, ipix_clean_cats, nside, ra_str, dec_str, half_size_plot, output_dir):
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
    output_dir : str
        Folder where the plots will be saved.
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
    fig, ax = plt.subplots(tot_clus, 3, figsize=(12, 4*tot_clus))
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
            line = j
            data = fits.getdata(ipix_clean_cats[i])
            RA = data[ra_str]
            DEC = data[dec_str]
            col = 0
            ax[line, col].scatter(
                RA_orig, DEC_orig, edgecolor='b', color='None', s=20, label='All stars')
            ax[line, col].set_xlim(
                [ra_cen[i] + half_size_plot, ra_cen[i] - half_size_plot])
            ax[line, col].set_ylim(
                [dec_cen[i] - half_size_plot, dec_cen[i] + half_size_plot])
            ax[line, col].set_title('Ipix='+str(ipix[i]), y= 0.9, pad=8) #{x=ra_cen[i], y=dec_cen[i], pad=8)
            ax[line, col].legend(loc=3)

            col = 1
            ax[line, col].scatter(RA, DEC, edgecolor='b', color='None', s=20, label='Filtered stars')
            ax[line, col].set_xlim(
                [ra_cen[i] + half_size_plot, ra_cen[i] - half_size_plot])
            ax[line, col].set_ylim(
                [dec_cen[i] - half_size_plot, dec_cen[i] + half_size_plot])
            ax[line, col].set_title('Ipix='+str(ipix[i]), y= 0.9, pad=8) #{x=ra_cen[i], y=dec_cen[i], pad=8)
            ax[line, col].legend(loc=3)
            
            col = 2
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
    plt.savefig(output_dir + '/clusters_with_and_without_crowded_stars.png')
    plt.show()
    plt.close()

    
def general_plots(star_clusters_simulated, output_dir):
    
    output_plots = Path(output_dir)
    output_plots.mkdir(parents=True, exist_ok=True)
    
    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()

    PIX_sim, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(
        star_clusters_simulated,
        usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15),
        unpack=True,
    )
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    ax1.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V[MAG_ABS_V < 0.0], color='r', label='Sim')
    ax1.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0], color='darkred', label='Sim filt')
    ax1.scatter(rhl_pc_DG, Mv_DG, color='b', marker='x', label='DG')
    ax1.scatter(rhl_pc_GC, Mv_GC, color='k', marker='x', label='GC')
    for i, j in enumerate(R_EXP):
        if MAG_ABS_V[i] < 0.0:
            ax1.plot([1.7 * R_EXP[i], 1.7 * R_EXP[i]],
                     [MAG_ABS_V[i], MAG_ABS_V_CLEAN[i]], color='darkred', lw=0.1)
    for i, j in enumerate(rhl_pc_DG):
        ax1.annotate(name_DG[i], (rhl_pc_DG[i], Mv_DG[i]))
    for i, j in enumerate(rhl_pc_GC):
        ax1.annotate(name_GC[i], (rhl_pc_GC[i], Mv_GC[i]))
    ax1.set_ylabel("M(V)")
    ax1.set_xlabel(r"$r_{1/2}$ (pc))")
    ax1.set_xlim([np.min(1.7 * R_EXP[MAG_ABS_V < 0.0]) - 0.1, np.max(1.7 * R_EXP[MAG_ABS_V < 0.0]) + 0.1])
    ax1.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) + 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    ax1.set_xscale("log")
    ax1.legend()

    ax2.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V[MAG_ABS_V < 0.0], color='r', label='Sim')
    ax2.scatter(1.7 * R_EXP[MAG_ABS_V < 0.0], MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0], color='darkred', label='Sim filt')
    ax2.scatter(rhl_pc_DG, Mv_DG, color='b', marker='x', label='DG')
    ax2.scatter(rhl_pc_GC, Mv_GC, color='k', marker='x', label='GC')
    #for i, j in enumerate(rhl_pc_DG):
    #    ax2.annotate(name_DG[i], (np.log10(rhl_pc_DG[i]), Mv_DG[i]))
    #for i, j in enumerate(rhl_pc_GC):
    #    ax2.annotate(name_GC[i], (np.log10(rhl_pc_GC[i]), Mv_GC[i]))
    ax2.set_xlabel(r"$r_{1/2}$ (pc))")
    ax2.legend()
    ax2.plot(np.logspace(np.log10(1.8), np.log10(1800), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.plot(np.logspace(np.log10(4.2), np.log10(4200), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.plot(np.logspace(np.log10(11), np.log10(11000), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.plot(np.logspace(np.log10(28), np.log10(28000), 10, endpoint=True),
        np.linspace(1, -14, 10, endpoint=True), color="b", ls=":")
    ax2.text(300, -7.9, r"$\mu_V=27\ mag/arcsec$", rotation=45)
    ax2.text(400, -4.2, r"$\mu_V=31\ mag/arcsec$", rotation=45)
    ax2.set_xscale("log")
    ax2.set_xlim([0.4, 4000])
    ax2.set_ylim([1, -14])

    ax3.scatter(MASS, MAG_ABS_V, label='Sim', color='r')
    ax3.scatter(MASS, MAG_ABS_V_CLEAN, label='Sim filt', color='darkred')
    for i, j in enumerate(MASS):
        if MAG_ABS_V[i] < 0.0:
            ax3.plot([MASS[i], MASS[i]],
                     [MAG_ABS_V[i], MAG_ABS_V_CLEAN[i]], color='darkred', lw=0.2)
    ax3.set_xlabel("mass(Msun)")
    ax3.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) + 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    ax3.legend()
    plt.savefig(output_dir + '/hist_MV.png')
    plt.show()
    plt.close()


def plot_ftp(
    ftp_fits, star_clusters_simulated, mockcat, ra_max, ra_min, dec_min, dec_max,
    output_dir
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
    plt.clf()

    # TODO: Verificar variaveis carregadas e não usadas
    PIX_sim, NSTARS, MAG_ABS_V, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(
        star_clusters_simulated,
        usecols=(0, 1, 2, 9, 10, 11, 12, 13, 14, 15),
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
    axs.scatter(RA, DEC, s=0.01, c="k", marker="s", label="Simulated clusters")
    axs.scatter(RA_star, DEC_star, s=0.01, c="k", marker="o", label="Simulated stars")
    axs.set_xlim([ra_max, ra_min])
    axs.set_ylim([dec_min, dec_max])
    axs.set_xlabel("RA (deg)")
    axs.set_ylabel("DEC (deg)")
    axs.set_title("Distribution of stars on Footprint Map")
    axs.grid()
    plt.legend(loc=1)
    plt.savefig(output_dir + '/ftp.png')
    plt.show()
    plt.close()


def plots_ang_size(
    star_clusters_simulated,
    clus_path,
    mmin,
    mmax,
    cmin,
    cmax,
    output_plots
):
    """Plots to analyze the simulated clusters."""

    cmap = mpl.cm.get_cmap("inferno")
    cmap.set_under("dimgray")
    cmap.set_bad("black")

    # TODO: Variaveis Instanciadas e não usadas
    hp_sample_un, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, RA_pix, DEC_pix, r_exp, ell, pa, mass, dist = np.loadtxt(
        star_clusters_simulated, usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15), unpack=True
    )

    for i in hp_sample_un:

        # TODO: use only cats of filtered stars
        clus_filepath = Path(clus_path, "%s_clus.dat" % int(i))
        plot_filepath = Path(output_plots, "%s_cmd.png" % int(i))
        plot_filt_filepath = Path(output_plots, "%s_filt_cmd.png" % int(i))

        # Evita erro se o arquivo _clus.dat não existir.
        # Pode acontecer se o teste estiver usando uma quantidade pequena de dados.
        if not clus_filepath.exists():
            continue

        star = np.loadtxt(clus_filepath)

        plt.scatter(star[:,2]-star[:,4], star[:,2], color='r')
        plt.title('HPX ' + str(int(i)) + ', N=' + str(len(star[:,2])))
        plt.ylim([mmax, mmin])
        plt.xlim([cmin, cmax])
        plt.xlabel('mag1-mag2')
        plt.ylabel('mag1')
        plt.savefig(str(int(i)) + '_cmd.png')
        plt.show()
        plt.close()

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
        plt.show()
        plt.close()
        
    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()
    
    ang_size_DG = 60. * (180. / np.pi) * np.arctan(rhl_pc_DG / (1000. * dist_kpc_DG))
    ang_size = 60 * np.rad2deg(np.arctan(1.7 * r_exp / dist))
    
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(22, 20))

    ax1.hist(dist / 1000, bins=np.linspace(np.min(dist) / 2000, 2. * np.max(dist) / 1000, 20), label='Sim', color='r', alpha=0.5)
    ax1.hist(dist_kpc_DG, bins=np.linspace(np.min(dist) / 2000, 2. * np.max(dist) / 1000, 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax1.hist(dist_kpc_GC, bins=np.linspace(np.min(dist) / 2000, 2. * np.max(dist) / 1000, 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax1.legend()
    ax1.set_xlabel("Distance (kpc)")
    ax1.set_ylabel("N objects")
    ax1.set_yscale('log')
    #plt.savefig(output_plots + '/hist_dist_log.png')
    #plt.show()
    #plt.close()

    ax2.hist(dist / 1000, bins=np.linspace(np.min(dist) / 2000, 2. * np.max(dist) / 1000, 20), label='Sim', color='r', alpha=0.5)
    ax2.hist(dist_kpc_DG, bins=np.linspace(np.min(dist) / 2000, 2. * np.max(dist) / 1000, 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax2.hist(dist_kpc_GC, bins=np.linspace(np.min(dist) / 2000, 2. * np.max(dist) / 1000, 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax2.legend()
    ax2.set_xlabel("Distance (kpc)")
    ax2.set_ylabel("N objects")
    #plt.savefig(output_plots + '/hist_dist.png')
    #plt.show()
    #plt.close()

    ax3.hist(ang_size, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='Sim', color='r', alpha=0.5)
    ax3.hist(ang_size_DG, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax3.hist(rhl_arcmin_GC, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_xlabel("Half-light radii (arcmin)")
    ax3.set_ylabel("N objects")
    # plt.savefig(output_plots + '/hist_ang_size_log.png')
    # plt.show()
    # plt.close()

    ax4.hist(ang_size, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='Sim', color='r', alpha=0.5)
    ax4.hist(ang_size_DG, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='DG', color='b', alpha=0.5, histtype='stepfilled')
    ax4.hist(rhl_arcmin_GC, bins=np.linspace(np.min(ang_size) / 2, 2. * np.max(ang_size), 20), label='GC', color='k', alpha=0.5, lw=2, histtype='step')
    ax4.legend()
    ax4.set_xlabel("Half-light radii (arcmin)")
    ax4.set_ylabel("N objects")
    #plt.savefig(output_plots + '/hist_ang_size.png')
    #plt.show()
    #plt.close()

    ax5.scatter(dist / 1000, ang_size, label='Sim', color='r')
    ax5.scatter(dist_kpc_DG, ang_size_DG, label='DG', color='b')
    ax5.scatter(dist_kpc_GC, rhl_arcmin_GC, label='GC', color='k')
    ax5.set_xlabel("Distance (kpc)")
    ax5.set_ylabel("Half-light radii (arcmin)")
    ax5.set_yscale('log')
    ax5.legend()
    # plt.savefig(output_plots + '/rhl_versus_dist.png')
    # plt.show()
    # plt.close()
    
    for i, j in enumerate(mass):
        if MAG_ABS_V[i] < 0.0:
            ax6.plot([mass[i], mass[i]],
                     [NSTARS[i], NSTARS_CLEAN[i]], color='darkred', lw=0.2)
    ax6.scatter(mass, NSTARS, label='Sim', color='r')
    ax6.scatter(mass, NSTARS_CLEAN, label='Sim filt', color='darkred')
    ax6.set_xlabel("MASS(MSun)")
    ax6.set_ylabel("N stars")
    ax6.legend()
    # plt.savefig(output_plots + '/hist_mass.png')
    plt.show()
    # plt.close()

    '''
    plt.scatter(mass, MAG_ABS_V, label='Sim', color='r')
    plt.scatter(mass, MAG_ABS_V_CLEAN, label='Sim filt', color='darkred')
    plt.xlabel("MASS(MSun)")
    plt.ylabel("MAG_ABS_V")
    plt.set_ylim([np.max(MAG_ABS_V_CLEAN[MAG_ABS_V < 0.0]) + 0.1, np.min(MAG_ABS_V[MAG_ABS_V < 0.0]) - 0.1])
    plt.legend()
    plt.savefig(output_plots + '/mass_abs_mag.png')
    plt.show()
    plt.close()
    '''


def plots_ref(FeH_iso,
              output_plots,
              star_clusters_simulated=Path("results/star_clusters_simulated.dat"), 
):
    """Make a few plots about the simulated clusters"""

    name_DG, ra_DG, dec_DG, dist_kpc_DG, Mv_DG, rhl_pc_DG, FeH_DG, name_GC, R_MW_GC, FeH_GC, mM_GC, Mv_GC, rhl_pc_GC, dist_kpc_GC, rhl_arcmin_GC = read_real_cat()
    
    # Star Clusters Simulated
    # TODO: Variaveis instanciadas e não utilizadas
    star_clusters_simulated = Path(star_clusters_simulated)

    PIX_sim, NSTARS, MAG_ABS_V, NSTARS_CLEAN, MAG_ABS_V_CLEAN, RA, DEC, R_EXP, ELL, PA, MASS, DIST = np.loadtxt(
    star_clusters_simulated,
    usecols=(0, 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15),
    unpack=True)

    LOG10_RHL_PC_SIM = np.log10(1.7 * R_EXP)

    MW_center_distance_DG_kpc = radec2GCdist(ra_DG, dec_DG, dist_kpc_DG)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs[0, 0].hist(
        MAG_ABS_V,
        bins=20,
        range=(-16, 0.0),
        histtype="step",
        label="Sim",
        color="r",
        ls="--",
        alpha=0.5
    )
    axs[0, 0].hist(
        MAG_ABS_V_CLEAN,
        bins=20,
        range=(-16, 0.0),
        histtype="stepfilled",
        label="Sim filt",
        color="darkred",
        ls="--",
        alpha=0.5
    )
    
    axs[0, 0].hist(
        Mv_DG, bins=20, range=(-16, 0.0), histtype="stepfilled", label="DG", color="b", alpha=0.5
    )
    axs[0, 0].hist(
        Mv_GC, bins=20, range=(-16, 0.0), histtype="step", label="GC", color="k"
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
        color="r",
        ls="--",
        alpha=0.5
    )
    axs[0, 1].hist(
        np.log10(rhl_pc_DG),
        bins=20,
        histtype="stepfilled",
        range=(0, 4.0),
        label="DG",
        color="b",
        alpha=0.5
    )
    axs[0, 1].hist(
        np.log10(rhl_pc_GC),
        bins=20,
        histtype="step",
        range=(0, 4.0),
        label="GC",
        color="k",
    )
    axs[0, 1].set_xlabel("log10(rhl[pc])")
    axs[0, 1].legend(loc=1)

    axs[1, 0].hist(
        DIST / 1000,
        bins=20,
        range=(0, 400.0),
        histtype="stepfilled",
        label="Sim",
        color="r",
        ls="--",
        alpha=0.5
    )
    axs[1, 0].hist(
        dist_kpc_DG, bins=20, range=(0, 400.0), histtype="stepfilled", label="DG", color="b", alpha=0.5
    )
    axs[1, 0].hist(
        dist_kpc_GC, bins=20, range=(0, 400.0), histtype="step", label="GC", color="k"
    )
    axs[1, 0].set_xlabel("Distance (kpc)")
    axs[1, 0].legend(loc=1)

    axs[1, 1].hist(
        np.repeat(FeH_iso, len(MAG_ABS_V)),
        bins=20,
        range=(-3, 1.0),
        histtype="stepfilled",
        label="Sim",
        color="r",
        ls="--",
        alpha=0.5
    )
    axs[1, 1].hist(
        FeH_DG, bins=20, range=(-3, 1.0), histtype="stepfilled", label="DG", color="b", alpha=0.5
    )
    axs[1, 1].hist(
        FeH_GC, bins=20, range=(-3, 1.0), histtype="step", label="GC", color="k"
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
    plt.show()
    plt.close()

    #  PLOT 2 ----------------
    plt.scatter(
        DIST / 1000,
        np.repeat(FeH_iso, len(DIST)),
        label="Sim",
        color="r",
        marker="x",
        lw=1.0,
    )
    plt.scatter(MW_center_distance_DG_kpc, FeH_DG, label="DG", color="b")
    plt.scatter(R_MW_GC, FeH_GC, label="GC", color="k")
    plt.xlabel("Distance to the Galactic center (kpc)")
    plt.ylabel("[Fe/H]")
    plt.ylim([-3.5, 0])
    plt.legend()
    plt.grid()
    filepath = Path(output_plots, "_02_feh_rgc.png")
    plt.savefig(filepath)
    plt.show()
    plt.close()

    # rhl = np.logspace(np.log10(1.8), np.log10(1800), 10, endpoint=True)
    # m_v = np.linspace(1, -14, 10, endpoint=True)

    '''
    #  PLOT 3 ----------------
    plt.scatter(1.7 * R_EXP, MAG_ABS_V, marker="s", color="r", label="Sim")
    plt.scatter(1.7 * R_EXP, MAG_ABS_V_CLEAN, marker="s", color="darkred", label="Sim filt")
    plt.scatter(rhl_pc_DG, Mv_DG, marker="x", color="b", label="DG")
    plt.scatter(rhl_pc_GC, Mv_GC, marker="x", color="k", label="GC")
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
    plt.show()
    plt.close()
    '''

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

    # TODO: Variaveis instanciadas mas não utilizadas
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
    plt.close()
