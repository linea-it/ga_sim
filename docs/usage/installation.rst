============
Installation
============

| To install the code, you have to clone the repository and
| follow instructions on README to install dependencies.

| The code is intended to run on Linux OS.


Code and dependencies
=====================

| The code is written and runs in Python 3.10.4, but it is compatible
| to python 3.X. The following libraries are mandatory to run the code:

* `numpy <https:/numpy.org/>`_
* `astropy <https:/www.astropy.org/>`_
* `pathlib <https:/docs.python.org/3/library/pathlib.html>`_
* `healpy <https:/healpy.readthedocs.io/en/latest>`_
* `sqlalchemy <https:/www.sqlalchemy.org>`_
* `json <https:/docs.python.org/3/library/json.html>`_
* `os <https:/docs.python.org/3/library/os.html>`_
* `sys <https:/docs.python.org/3/library/sys.html>`_
* `glob <https:/docs.python.org/3/library/glob.html>`_
* `parsl <https:/parsl-project.org/>`_
* `tqdm <https:/tqdm.github.io/>`_
* `time <https:/docs.python.org/3/library/time.html>`_
* `matplotlib <https:/matplotlib.org/>`_
* `collections <https:/docs.python.org/3/library/collections.html>`_
* `warnings <https:/docs.python.org/3/library/warnings.html>`_
* `scipy <https:/scipy.org/>`_
* `itertools <https:/docs.python.org/3/library/itertools.html>`_
* `tabulate <https:/pypi.org/project/tabulate/>`_


Installation
============

| Clone the repository and create an environment with Conda:

::

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

| Once you created this env, in the second time (and after)
| you run the code, you can only access the env activating it:

::

	conda activate ga_sim


| If you have error messages from missing libraries,
| install it in a similar manner as packages installed above.


