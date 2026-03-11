#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

echo "x86" > test_compute_time.txt
conda activate dolfin_warp
echo CONDA_PREFIX=$CONDA_PREFIX
unset OMP_NUM_THREADS
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
rm -rf $CONDA_PREFIX/.cache/dijitso
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
conda deactivate

echo "x86 (OMP_NUM_THREADS=1)" >> test_compute_time.txt
conda activate dolfin_warp
echo CONDA_PREFIX=$CONDA_PREFIX
export OMP_NUM_THREADS=1
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
rm -rf $CONDA_PREFIX/.cache/dijitso
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
conda deactivate

echo "arm" >> test_compute_time.txt
conda activate dolfin_warp_arm
echo CONDA_PREFIX=$CONDA_PREFIX
unset OMP_NUM_THREADS
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
rm -rf $CONDA_PREFIX/.cache/dijitso
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
conda deactivate
