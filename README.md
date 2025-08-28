# wsi-cluster

A reproducible pipeline for **self-supervised learning (SSL) feature embeddings** on whole slide image (WSI) patches.  
This repo trains SSL models (SimCLR, MoCo, DINO, DenseCL) using PyTorch Lightning + Lightly, then extracts embeddings for downstream clustering or visualization.

---
## Environment Setup

This repo uses a **Python venv** inside an **Apptainer image**.  

This setup will require an interactive job on CCR.

```bash
#Request with salloc command 
salloc --clusters=faculty --partition=sunycell --qos=sunycell --mem=50G --nodes=1 --time=6:00:00 --ntasks-per-node=1 --gpus-per-node=2 --cpus-per-task=32

#After resources have been allocated, insert given jobid
srun --jobid=JOBID_HERE --export=HOME,TERM,SHELL --pty /bin/bash --login
```

Download the pytorch container from [NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)  
To avoid quota issues, set your cache directory to your *current* interactive slurm job's cache. 
```bash
# change cache dir
export APPTAINER_CACHEDIR=$SLURMTMPDIR

#pull the image
apptainer pull <localname>.sif <container-registry>://<repository-name>
```

Run an interactive 


Navigate to wsi-cluster root, then:  

```bash
# Create a virtual environment (while in wsi-cluster root)
python3 -m venv wsi-cluster
source wsi-cluster/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt --no-cache-dir
```

## Training an SSL Model
Be sure to check the following, edit lightly_scripts/train_ssl.slurm as necessary:
  * Paths to data
  * Output directory paths
  * Any other paths in train_ssl.slurm
  * Resources and allocations on slurm script
  * Hyperparameters
  

Submit the job with the following (from the wsi-cluster root):
```bash
sbatch lightly_scripts/train_ssl.slurm
```

## Extracting Features
Checkpoints are saved to: <output_dir>/<run_name>/  

**Required**: Point extract_features.slurm to your desired .ckpt file  

Be sure of the following, edit lightly_scripts/extract_features.slurm as necessary:
 * Method must match the method used in training
 * Check DATA_DIR/IMAGE_FOLDER, this should match as well
 * OUTPUT_ROOT --> where embeddings should be stored

Run:  

```bash
sbatch lightly_scripts/extract_features.slurm
```

This will produce two things:
 * Master .pt file containing the embeddings for all patches  
 * Individual .pt files for each individual patch  


