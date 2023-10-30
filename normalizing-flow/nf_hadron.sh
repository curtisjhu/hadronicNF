#!/bin/bash
#SBATCH -A m3443
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=curtisjhu@berkeley.edu
#SBATCH -L scratch

export SLURM_CPU_BIND="cores"

module load cudnn
srun python ./gan4hep/nf/train_cond_hadronic.py --config_file "./gan4hep/nf/config_nf_hadronic.yml" --log-dir "./train_out/pi_mode_20_bijectors" --data-dir "/global/cfs/cdirs/m3443/data/ForHadronic/train_data/pimode/pimode.hkl" --epochs 250