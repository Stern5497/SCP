#!/bin/bash
#SBATCH --job-name="LEXTREME"
#SBATCH --mail-user=ronja.stern@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu
#SBATCH --array=2-4

# enable this for multiple GPUs for max 24h
###SBATCH --time=24:00:00
###SBATCH --qos=job_gpu_preempt
###SBATCH --partition=gpu

# enable this to get a time limit of 20 days
###SBATCH --time=20-00:00:00
###SBATCH --qos=job_gpu_stuermer
###SBATCH --partition=gpu-invest

# load modules
module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.3.0-GCC-10.2.0

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate scp-test

# Put your code below this line

# python main.py -t swiss_bge_criticality_prediction -od results/final_results -los ${SLURM_ARRAY_TASK_ID} --num_train_epochs 10

python main.py --task swiss_bge_criticality_prediction -gm 24 -bz 8 -los 1 --num_train_epochs 10 -ld results/test_scp --language_model_type microsoft/mdeberta-v3-base --hierarchical True --running_mode experimental

# python main.py --task swiss_bge_criticality_prediction -los 1 --num_train_epochs 10 -od results/test_scp --language_model_type distilbert-base-multilingual-cased --hierarchical True

# IMPORTANT:
# Run with                  sbatch run_hpc_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --cpus-per-task=8 --time=02:00:00 --pty /bin/bash
