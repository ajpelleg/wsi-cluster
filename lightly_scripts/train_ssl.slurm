#!/bin/bash -l
#
#SBATCH --clusters=faculty
#SBATCH --partition=sunycell
#SBATCH --qos=sunycell

#SBATCH --mem=500000
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=cpn-v09-38
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00

#SBATCH --job-name="ssl_train"
#SBATCH --output=logs/ssl_train_%j.log
#SBATCH --mail-user=ajpelleg@buffalo.edu
#SBATCH --mail-type=ALL
#SBATCH --requeue

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOBID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs/node: $SLURM_GPUS_ON_NODE"
echo "CPUs/task: $SLURM_CPUS_PER_TASK"
echo "================"

# Defaults (overridable via sbatch flags)
METHOD=${METHOD:-simclr}
BACKBONE=${BACKBONE:-resnet50}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-100}
LR=${LR:-}             # if empty, train_ssl.py computes default
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
VAL_SPLIT=${VAL_SPLIT:-0.2}
NUM_WORKERS=${NUM_WORKERS:-8}

DATA_DIR="/PathLDM/features/data"
IMAGE_FOLDER="tumor_segmentation_v2_05mpp_256/tiles/images"
OUTPUT_DIR="/PathLDM/experiments/${METHOD}"


echo "=== Hyperparameters ==="
echo "Method:        $METHOD"
echo "Backbone:      $BACKBONE"
echo "Batch size:    $BATCH_SIZE"
echo "Epochs:        $EPOCHS"
echo "Learning rate: ${LR:-(auto)}"
echo "Weight decay:  $WEIGHT_DECAY"
echo "Val split:     $VAL_SPLIT"
echo "Workers:       $NUM_WORKERS"
echo "======================="

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

echo "Launching training inside Apptainer..."

apptainer exec --nv \
  -B /projects/academic/sunycell/PathLDM:/PathLDM \
  /projects/academic/sunycell/apptainers/pytorch26.sif \
  bash -lc "\
    cd /PathLDM/lightly_scripts && \
    source /PathLDM/miniconda/etc/profile.d/conda.sh && \
    conda activate lightly_train && \
    python train_ssl.py \
      --method        $METHOD \
      --backbone      $BACKBONE \
      --batch_size    $BATCH_SIZE \
      --epochs        $EPOCHS \
      --lr            $LR \
      --weight_decay  $WEIGHT_DECAY \
      --val_split     $VAL_SPLIT \
      --num_workers   $NUM_WORKERS \
      --data_dir      $DATA_DIR \
      --image_folder  $IMAGE_FOLDER \
      --output_dir    $OUTPUT_DIR \
      --gpus          1 \
      --save_top_k    1

  "
echo "Training complete."


