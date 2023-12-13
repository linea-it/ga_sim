export CONDAPATH=/lustre/t1/cl/lsst/gawa_project/adriano.pieres/conda/bin
export GA_SIM_ROOT=/lustre/t1/cl/lsst/gawa_project/adriano.pieres/ga_sim/ga_sim
export PYTHONPATH=$PYTHONPATH:$GA_SIM_ROOT

source $CONDAPATH/activate
conda activate /lustre/t1/cl/lsst/gawa_project/adriano.pieres/conda/envs/ga_sim
python -m ipykernel install --user --name=ga_sim
