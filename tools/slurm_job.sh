#!/bin/bash
#SBATCH -p your_partition -l40s # partition (queue)
#SBATCH -t 1-00:00            # time (D-HH:MM)
#SBATCH --gres=gpu:8          # number of GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
##SBATCH --nice=999
#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
##SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts)
##SBATCH --exclude some_node_you_exclude   # Exclude nodes

#set -x # This will print the commands to the output, can be useful for debugging

CONFIG=$1
RESUME=$2
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

# Raise error if job name is not provided
if [ -z "${SLURM_JOB_NAME}" ]; then
    echo "Job name not provided. Exiting."
    exit 1
fi
# Raise error if config file is not provided
if [ -z "${CONFIG}" ]; then
    echo "Config file not provided. Exiting."
    exit 1
fi

# Set up distributed environment variables
export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=29500
# Choose random integer MASTER_PORT between 29500 and 29600
export MASTER_PORT=$(shuf -i 29000-30000 -n 1)
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL
#export NCCL_P2P_DISABLE=1

echo "Working directory is $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE CPUs and $SLURM_JOB_GPUS GPUs with job ID
$SLURM_JOB_ID on queue $SLURM_JOB_PARTITION. GPUS_PER_NODE: $SLURM_GPUS_PER_NODE, GPUS: $SLURM_GPUS, CPUS_PER_TASK: $SLURM_CPUS_PER_TASK,
SLURM_NTASKS_PER_NODE $SLURM_NTASKS, SRUN_ARGS: $SUN_ARGS";

source /etc/cuda_env
cuda11.7 # Set CUDA version

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/miniconda3/bin/activate dualviewmapdet # Adjust to your path of Miniconda installation

# Running the job
start=`date +%s`

PYTHONPATH="./":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${SLURM_JOB_NAME} \
    --gres=gpu:${SLURM_NTASKS} \
    --ntasks=${SLURM_NTASKS} \
    --ntasks-per-node=${SLURM_NTASKS} \
    --cpus-per-task=${SLURM_CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" $RESUME ${PY_ARGS}

echo "Job ID: ${SLURM_JOB_ID}"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
