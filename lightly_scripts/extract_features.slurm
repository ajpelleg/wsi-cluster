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

# these can be overridden later with sbatch export
CKPT=${CKPT:-"/wsi-cluster/experiments/checkpoints/dino_final.ckpt"}
METHOD=${METHOD:-"dino"}
DATA_DIR=${DATA_DIR:-"/features/data"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"tumor_segmentation_v2_05mpp_256/tiles/images"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/wsi-cluster/experiments/features"}

echo "Checkpoint:   $CKPT"
echo "Method:       $METHOD"
echo "Data Dir:     $DATA_DIR"
echo "ImageFolder:  $IMAGE_FOLDER"
echo "BatchSize:    $BATCH_SIZE"
echo "NumWorkers:   $NUM_WORKERS"
echo "OutputRoot:   $OUTPUT_ROOT"
echo

apptainer exec --nv \
  -B /wsi-cluster:/wsi-cluster \
  /apptainers/pytorch26.sif \
bash -lc "
  cd /wsi-cluster/lightly_scripts
  source /wsi-cluster/miniconda/etc/profile.d/conda.sh
  conda activate lightly_train

  python extract_features.py \
    \"$CKPT\" \
    --method        $METHOD \
    --data_dir      \"$DATA_DIR\" \
    --image_folder  \"$IMAGE_FOLDER\" \
    --batch_size    $BATCH_SIZE \
    --num_workers   $NUM_WORKERS \
    --output_root   \"$OUTPUT_ROOT\"
"

echo "Extraction finished for $METHOD"
