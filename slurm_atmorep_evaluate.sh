#!/bin/bash -x
#SBATCH --account=ehpc03
#SBATCH --time=0-0:09:59
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80 #####nodes * gpus/node * 20
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --qos=acc_ehpc
#SBATCH --output=logs/atmorep-%x.%j.out
#SBATCH --error=logs/atmorep-%x.%j.err

# import modules
source pyenv/bin/activate

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=TRACE
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

echo "Starting job."
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
date

CONFIG_DIR=${SLURM_SUBMIT_DIR}/atmorep_eval_${SLURM_JOBID}
mkdir ${CONFIG_DIR}
cp ${SLURM_SUBMIT_DIR}/atmorep/core/evaluate.py ${CONFIG_DIR}
echo "${CONFIG_DIR}/train.py"
srun --label --cpu-bind=v ${SLURM_SUBMIT_DIR}/pyenv/bin/python -u ${CONFIG_DIR}/evaluate.py > output/output_${SLURM_JOBID}.txt

echo "Finished job."
date