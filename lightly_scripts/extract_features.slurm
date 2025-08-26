#!/bin/bash -l
#
#SBATCH --cluster=faculty
#SBATCH --partition=sunycell
#SBATCH --qos=sunycell
#SBATCH --mem=500000
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00

#SBATCH --job-name="extract_feats"
#SBATCH --output=logs/extraction/extract_feats_%j.log
#SBATCH --requeue

echo "=== Job Info ==="
echo " JobID:    $SLURM_JOBID"
echo " NodeList: $SLURM_JOB_NODELIST"
echo " GPUs/Node:$SLURM_GPUS_ON_NODE"
echo " CPUs/Task:$SLURM_CPUS_PER_TASK"
echo

VENV_DIR="$HOME/.venvs/wsi-cluster"

CKPT="/wsi-cluster/experiments/dino/lightning_logs/.ckpt # <- set me
METHOD="dino"                   # simclr|moco|dino|densecl
DATA_DIR="/wsi-cluster/data"
IMAGE_FOLDER="tumor_segmentation_v2_05mpp_256/tiles/images"
BATCH_SIZE=64
OUTPUT_ROOT="/wsi-cluster/results"

echo "Checkpoint:   $CKPT"
echo "Method:       $METHOD"
echo "Data Dir:     $DATA_DIR"
echo "ImageFolder:  $IMAGE_FOLDER"
echo "BatchSize:    $BATCH_SIZE"
echo "NumWorkers:   $NUM_WORKERS"
echo "OutputRoot:   $OUTPUT_ROOT"
echo

mkdir -p logs/extraction "${OUTPUT_ROOT}"

source "${VENV_DIR}/bin/activate"

srun python -u extract_features.py \
  "${CKPT}" \
  --method        "${METHOD}" \
  --data_dir      "${DATA_DIR}" \
  --image_folder  "${IMAGE_FOLDER}" \
  --batch_size    ${BATCH_SIZE} \
  --num_workers   ${NUM_WORKERS} \
  --output_root   "${OUTPUT_ROOT}"

echo "Extraction finished."
