#<div id="top"></div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Galactic Archaeology Simulations (ga_sim)

The code in this repository is aimed to be run on the LIneA environment (simple python scripts to run on cluster and on jupyter noteboook environment). The code written as jupyter notebooks are intended to produce html pages in order to have a Quality Assurement about the simulations.

The code is aimed to produce simulations of dwarf galaxies and stellar clusters embedded in a catalog. A few files are also needed to run the code and its paths are indicated on the code. All the files needed to run the code are incorporated in the repository. If you detect a missing file, please let us know.

This project is intended to create a complete catalog of field stars and simulated stellar systems. The field stars are read from a catalog (currently a version of DES catalog or a simulation for LSST) in Linea's DB. The position of the field stars (from previous catalogs) are randomized in order to avoid real objects (stellar overdensities) in background. The limits for the position of the stars are read from footprint map. Simulated clusters are created based on an
isochronal model (each time the code is run an isochrone is downloaded such a way that the code is using the most updated model) and later inserted on the catalog along with the field stars. All these parts are running on a python script, were the most time-consuming functions runs in parallel (basically the creation of stellar substructures and the observational selection function). Each simulated cluster is centered on a HealPix pixel, with nside defined by user. The configuration file also sets sizes, mass, distances and many features (or minimum and maximum features) to the simulated clusters. All the code to simulate clusters runs on a python script in order to get advantage of the cluster of computers. In the end of the code many plots are
produced to check distribution of clusters and field stars regarding absolute magnitude,
sizes, distances, etc. The final jupyter notebook is exported as an html page to be evaluated by the user.

The tables with isochrone's data were downloaded from [CMD web interface](http://stev.oapd.inaf.it/cgi-bin/cmd).

Currently the functions running in parallel are using parsl (see reference below).

## Code and dependencies

The code is written and runs in Python 3.10.4, but it is compatible to python 3.X. The following libraries are mandatory to run the code:

* [numpy](https://numpy.org/)
* [astropy](https://www.astropy.org/)
* [pathlib](https://docs.python.org/3/library/pathlib.html)
* [healpy](https://healpy.readthedocs.io/en/latest/)
* [sqlalchemy](https://www.sqlalchemy.org/)
* [json](https://docs.python.org/3/library/json.html)
* [os](https://docs.python.org/3/library/os.html)
* [sys](https://docs.python.org/3/library/sys.html)
* [glob](https://docs.python.org/3/library/glob.html)
* [parsl](https://parsl-project.org/)
* [tqdm](https://tqdm.github.io/)
* [time](https://docs.python.org/3/library/time.html)
* [matplotlib](https://matplotlib.org/)
* [collections](https://docs.python.org/3/library/collections.html)
* [warnings](https://docs.python.org/3/library/warnings.html)
* [scipy](https://scipy.org/)
* [itertools](https://docs.python.org/3/library/itertools.html)
* [tabulate](https://pypi.org/project/tabulate/)


<!-- ABOUT THE PROJECT -->
## About The Project

This is a LIneA project to simulate dwarf galaxies and stellar clusters, in order to later test pipelines to detect these stellar substructures in catalogs covering large areas of the sky.

People involved (alphabetic order): Adriano Pieres, Amanda Fassarela, Ana Clara de Paula Moreira, Cristiano Singulani, Cristophe Benoist, Luiz Nicolaci da Costa, Michel Aguena.


<!-- GETTING STARTED -->
## Getting Started

Whether you use a jupyter notebook or the python script here, an environment must be created using conda. The code here is intended to run on cluster of LInea, so a few changes should be implemented if you want to run on other environment.

If this is the first time you are running the code in a new environment, please create
a conda environment and install the dependencies, whether using `pip` or `conda`.
Be sure that these libraries are installed without errors. See the information on 'Installation' to a complete list of steps.

### Installation

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/ga_sim && cd ga_sim
conda create -p $HOME/.conda/envs/ga_sim python=3.8
conda activate ga_sim
conda install -c anaconda sqlalchemy
conda install -c anaconda psycopg2
conda install -c conda-forge tqdm
conda install jupyterlab
conda install ipykernel
pip install numpy
pip install tabulate
pip install astropy
pip install healpy
pip install --user parsl
ipython kernel install --user --name=ga_sim
```
Once you created this env, in the second time (and after) you can only access the env activating it:
```bash
conda activate ga_sim
```
If you have error messages from missing libraries, install it in a similar manner as packages installed above.

### Running

After activate the environment, run the code in terminal:
```bash
cd ~/ga_sim
python ga_sim.py
```
Be sure that you are in the same folder as the code cloned.

In case you want to run jupyter notebook:

```bash
jupyter-lab gawazpy.ipynb
```
Restart the jn kernel to load the libraries installed.
Run the jn cell by cell.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

The actions below are executed:

- Read config file (take a time to check all items in config file) and reddening maps;
- Download the isochrone following age and metalicity in cofig file. Check the connection if you have problems in downloading this file);
- Create a footprint based on limits in RA and DEC. See that limits of footprint follows geodesics in sky (TODO: change to Astropy Heapix instead of Healpy);
- Read catalogs and randomize position of stars in sky to avoid gradients and previous clustered stars. Stars are extinction corrected following the reddening maps;
- Generate a file table with all the features of simulated clusters: positions, masses, distances, exponential radius, ellipticity, angular position, etc (see complete list in code);
- Read photometric errors in specific file to survey (LSST errors came from a fit to DP0 stars; DES errors came from a fit to DES Y6 data);
- Creates (in parallel) a simulated cluster in each HealPixel following the sepcifications in file table of clusters. Each cluster is inserted in a file named with the i pixel. The photometric errors are taken into account to generate stars in simulated clusters;
- Join all the catalogs of stars into a single file with all the simulated stars. This step is optional;
- Use the catalog of stars in each pixel to apply observational bias (in parallel), removing stars that are closer than X arcsecs, in order to mimic the side-effect of group sources very close to each other (blending). In the code, if two sources are very close to each other, both are removed from catalog;
- Join all clear catalogs with stars filtered in;
- Writes features of stars after removing stars close to each other (recounting stars and recalculating absolute magnitude);
- Run a jupyter notebook to make plots and show results more confortably;


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Adriano Pieres - adriano.pieres@gmail.com

Cristiano Singulani - singulani@linea.gov.br

Project Link: [https://github.com/linea-it/ga_sim](https://github.com/linea-it/ga_sim)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [LInea-IT team](By its support)
* [Heising Simons Foundation](This work was supported by the Preparing for Astrophysics with LSST Program, funded by the Heising Simons Foundation through proposal KSI-10, and administered by Las Cumbres Observatory, during the period 2021-2023.
)

<p align="right">(<a href="#top">back to top</a>)</p>