#! /bin/bash
script_dir=/rds-d2/user/wjm41/rds-enaminereal-ZNFRY9wKoeE/EnamineREAL

# SCORING
target=$1
sbatch --job-name=gen_topN_$target \
       --output=${script_dir}/slurm_logs/subm_topN_${target}.out \
       --export=target=${target} subm_topN