#!/bin/bash
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3:00:00
#SBATCH --output=%j_train_b128_g1_slurm.out
#SBATCH --error=%j_train_b128_g1_slurm.err

# Change to the correct directory
cd x/

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load any necessary modules
# module load cuda/11.6.2

# Optional: print GPU information
nvidia-smi

# Run the training script
python -u train.py \
    --batch_size 128 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --log_interval 500

# Print end time
echo "End time: $(date)"
