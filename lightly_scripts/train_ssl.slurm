#!/bin/bash -l
#
#SBATCH --clusters=faculty
#SBATCH --partition=sunycell
#SBATCH --qos=sunycell

#SBATCH --mem=500000
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=cpn-v09-40
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

METHOD="dino"           # simclr | moco | dino | densecl
BACKBONE="resnet50"     # resnet18 | resnet50
BATCH_SIZE=64
EPOCHS=5
LR=5e-4
WEIGHT_DECAY=1e-6
VAL_SPLIT=0.2
SAVE_TOP_K=1
SEED=42
USE_OCC_SPLIT=1         #0 for regular random split

DATA_DIR="data"
IMAGE_FOLDER="tumor_segmentation_v2_05mpp_256/tiles/images"
OUTPUT_DIR="experiments/${METHOD}"


echo "=== Hyperparameters ==="
echo "Method:        $METHOD"
echo "Backbone:      $BACKBONE"
echo "Batch size:    $BATCH_SIZE"
echo "Epochs:        $EPOCHS"
echo "Learning rate: $LR"
echo "Weight decay:  $WEIGHT_DECAY"
echo "Val split:     $VAL_SPLIT"
echo "======================="


echo "Launching training inside Apptainer..."

apptainer exec --nv -B /projects/academic/sunycell/wsi-cluster:/wsi-cluster /projects/academic/sunycell/apptainers/pytorch26.sif \
  bash -lc "\
    cd /wsi-cluster && \
    source wsi-cluster/bin/activate && \
    python lightly_scripts/train_ssl.py \
      --method        $METHOD \
      --backbone      $BACKBONE \
      --batch_size    $BATCH_SIZE \
      --epochs        $EPOCHS \
      --lr            $LR \
      --weight_decay  $WEIGHT_DECAY \
      --val_split     $VAL_SPLIT \
      --data_dir      $DATA_DIR \
      --image_folder  $IMAGE_FOLDER \
      --output_dir    $OUTPUT_DIR \
      --gpus          1 \
      $( (( USE_OCC_SPLIT )) && echo --occ_split ) \
      --save_top_k    1

  "
echo "Training complete."


