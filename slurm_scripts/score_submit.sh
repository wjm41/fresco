#! /bin/bash
script_dir=/rds-d2/user/wjm41/rds-enaminereal-ZNFRY9wKoeE/EnamineREAL

# # SCORING
# target=$1
# sbatch --job-name=score_${target} \
#        --output=${script_dir}/slurm_logs/subm_score_${target}.out \
#        --export=target=${target} subm_score

# SCORING MPI
target=$1
sbatch --job-name=score_mpi_${target} \
       --output=${script_dir}/slurm_logs/subm_score_mpi_${target}.out \
       --export=target=${target} subm_score_mpi
