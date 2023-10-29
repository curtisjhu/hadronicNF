#!/bin/bash
#SBATCH -A m3443
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=achen8998@berkeley.edu
#SBATCH -L scratch

export SLURM_CPU_BIND="cores"

srun python /gan4hep/nf/train_cond_hadronic.py --config_file "/gan4hep/nf/config_nf_hadronic.yml" --log-dir "/pscratch/sd/a/achen899/train_out/pi_20_b_sorted" --data-dir "/pscratch/sd/a/achen899/train_data/pimode/pimode_sorted.hkl" --epochs 1200