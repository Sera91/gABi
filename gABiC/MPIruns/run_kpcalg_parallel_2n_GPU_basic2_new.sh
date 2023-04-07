#!/bin/bash
#SBATCH --job-name=kpc-dask-2node
#SBATCH --partition=gpu2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=10000
#SBATCH --time=01:00:00


module purge
module load gnu11 openmpi3  gsl cuda/11.0 #to modify


source /home/sdigioia/.bashrc

conda activate py310-ul


PATH="/home/sdigioia/R/bin:$PATH"
#LD_LIBRARY_PATH=/home/sdigioia/R/lib64/R/lib:${LD_LIBRARY_PATH}

export PATH=/home/sdigioia/R/bin:${PATH}
export LD_LIBRARY_PATH=/home/sdigioia/R/lib64/R/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=/home/sdigioia/R/lib64/pkgconfig/:${PKG_CONFIG_PATH}

#sleep 100

Nsample=(2000 5000)


for N in "${Nsample[@]}"
do

    cp params_basic_$N.py params_basic.py

    mpirun -n 6 python parallel_kpcalg_mpi_basic_dataset2.py > output_2nodes_HSIC_GPU_basic_dataset_${N}_new.txt

done

