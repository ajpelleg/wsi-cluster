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
#SBATCH --mail-user=ajpelleg@buffalo.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue

echo "=== Job Info ==="
echo " JobID:    $SLURM_JOBID"
echo " NodeList: $SLURM_JOB_NODELIST"
echo " GPUs/Node:$SLURM_GPUS_ON_NODE"
echo " CPUs/Task:$SLURM_CPUS_PER_TASK"
echo

# — override these with sbatch --export if you like — #
CKPT=${CKPT:-"/PathLDM/experiments/checkpoints/simclr_final.ckpt"}
METHOD=${METHOD:-"simclr"}
DATA_DIR=${DATA_DIR:-"/PathLDM/features/data"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"tumor_segmentation_v2_05mpp_256/tiles/images"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/PathLDM/experiments/features"}

echo "→ Checkpoint:   $CKPT"
echo "→ Method:       $METHOD"
echo "→ Data Dir:     $DATA_DIR"
echo "→ ImageFolder:  $IMAGE_FOLDER"
echo "→ BatchSize:    $BATCH_SIZE"
echo "→ NumWorkers:   $NUM_WORKERS"
echo "→ OutputRoot:   $OUTPUT_ROOT"
echo

# Make sure the host‐side dirs exist (matches the container’s /PathLDM/…)
HOST_OUTPUT=${HOST_OUTPUT:-"/projects/academic/sunycell/PathLDM${OUTPUT_ROOT#/PathLDM}"}
mkdir -p "$HOST_OUTPUT/$METHOD/individual"

apptainer exec --nv \
  -B /projects/academic/sunycell/PathLDM:/PathLDM \
  /projects/academic/sunycell/apptainers/pytorch26.sif \
bash -lc "
  cd /PathLDM/lightly_scripts
  source /PathLDM/miniconda/etc/profile.d/conda.sh
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
