#!/bin/bash

# Number of jobs you want to chain
job_name=$1  # Job name provided as the first argument
config=$2  # Config file provided as the second argument
n=${3:-5}  # Adjust this number to the desired number of jobs

# First job submission
jid=$(sbatch -J ${job_name} tools/slurm_job.sh ${config} ' ' | awk '{print $4}')
echo "Submitted job 1 with Job ID ${jid}"

# Submit subsequent jobs in a loop
for ((i=2; i<=n; i++))
do
    jid=$(sbatch --dependency=afterany:${jid} -J ${job_name} tools/slurm_job.sh ${config} '--resume' | awk '{print $4}')
    echo "Submitted job ${i} with Job ID ${jid}"
done

# Run slurm_eval_job.sh
#jid=$(sbatch --dependency=afterok:${jid} -J eval_${job_name} slurm_eval_job.sh ${config} | awk '{print $4}')
#echo "Submitted eval job with Job ID ${jid}"
