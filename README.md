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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Gawa Simulations

The code in this repository is aimed to be run on the LIneA jupyter notebooks. This is an ancillary code to produce simulations
and later be detected with Gawa detection code. A few files are also needed to run the code and its paths are indicated on the
code.

This project is intended to create a complete catalog of field stars and simulated stellar systems. The field stars are read from
a catalog (currently a version of DES catalog) in Linea's DB. The position of the field stars are randomized in order to avoid 
real objects (stellar overdensities) in background, and following the footprint map. Simulated clusters are created based on an
isochronal model (a few files with isochrones are added to the repository) and later inserted on the catalog. Many plots are
produced on the final part of the code to check the distribution of the clusters and field stars regarding absolute magnitude,
sizes, etc.

The tables with isochrone's data were downloaded from [CMD web interface](http://stev.oapd.inaf.it/cgi-bin/cmd).

The code is written in Python 3.X. The following libraries are mandatory to run the code:

* [os](https://docs.python.org/3/library/os.html)
* [healpy](https://healpy.readthedocs.io/en/latest/)
* [numpy](https://numpy.org/)
* [astropy](https://www.astropy.org/)
* [matplotlib](https://matplotlib.org/)
* [collections](https://docs.python.org/3/library/collections.html)
* [warnings](https://docs.python.org/3/library/warnings.html)
* [sqlalchemy](https://www.sqlalchemy.org/)
* [scipy](https://scipy.org/)
* [itertools](https://docs.python.org/3/library/itertools.html)
* [parsl](https://parsl-project.org/)


<!-- ABOUT THE PROJECT -->
## About The Project


This is a LIneA project to simulate dwarf galaxies and stellar clusters, in order to test pipelines to detect these
stellar substructures in catalogs.

People involved (alphabetic order): Adriano Pieres, Amanda Fassarela, Ana Clara de Paula Moreira, Cristiano Singulani,
Cristophe Benoist, Luiz Nicolaci da Costa.


<!-- GETTING STARTED -->
## Getting Started

To spawn the jupyter notebook code here, please use the last version of science image available on the
jupyter notebook facility on LIneA. Scipy and parsl are not installed on the last version of science, so you need
to open a terminal and install both packages along with pip:
```sh
python -m pip install -U --user pip
python -m pip install -U --user scikit-image
pip install --user parsl
```
Be sure that these libraries are installed without errors. See the information on 'Installation' to a complete list
of steps.

### Installation

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/ga_sim && cd ga_sim
conda create -p $HOME/.conda/envs/ga_sim python=3.8
conda activate ga_sim
conda install -c anaconda sqlalchemy
conda install -c anaconda psycopg2
conda install jupyterlab
conda install ipykernel
pip install numpy
pip install astropy
pip install healpy
ipython kernel install --user --name=ga_sim
```

### Running

```bash
jupyter-lab gawazpy.ipynb
```
Restart the jn kernel to load the libraries installed.
Run the jn cell by cell.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

In the first cells the code lists many inputs in order to set an initial configuration for field stars and clusters.
See the comments after the values in the code.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [] Feature 1
- [] Feature 2
- [] Feature 3
    - [] Nested Feature

This part should be significant to track the project and may be updated during the project.

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


<!-- ADDITINAL DOCUMENTATION -->
## Additional documentation

- [] Doc 1 - Description
- [] Doc 2 - Description
- [] Doc 3 - Description

This part may list additional detailed documentation (features, tests, diagrams about the project, etc).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Adriano Pieres - adriano.pieres@gmail.com

Cristiano Singulani - singulani@linea.gov.br

Project Link: [https://github.com/linea-it/gawa_simulations](https://github.com/linea-it/gawa_simulations)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>
