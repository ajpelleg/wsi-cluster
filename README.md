# wsi-cluster

A reproducible pipeline for **self-supervised learning (SSL) feature embeddings** on whole slide image (WSI) patches.  
This repo trains SSL models (SimCLR, MoCo, DINO, DenseCL) using PyTorch Lightning + Lightly, then extracts embeddings for downstream clustering or visualization.

---
## Clone this Github repository
```bash
git clone https://github.com/ajpelleg/wsi-cluster.git
```

## Environment Setup

This repo uses a **Python venv** inside an **Apptainer image**.  

This setup will require an interactive job on CCR.

```bash
#Request with salloc command 
salloc --clusters=faculty --partition=sunycell --qos=sunycell --mem=50G --nodes=1 --time=6:00:00 --ntasks-per-node=1 --gpus-per-node=2 --cpus-per-task=32

#After resources have been allocated, insert given jobid
srun --jobid=JOBID_HERE --export=HOME,TERM,SHELL --pty /bin/bash --login
```

Download the pytorch container from [NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).  
To avoid quota issues, set your cache directory to your *current* interactive slurm job's cache. 
```bash
# change cache dir
export APPTAINER_CACHEDIR=$SLURMTMPDIR

#pull the image
apptainer pull <localname>.sif <container-registry>://<repository-name>
```

Start an interactive apptainer session to setup up our **venv** and bind it to the wsi-cluster root folder.  
```bash
apptainer shell --nv -B <path_to_wsi-cluster_root>:/wsi-cluster <localname>.sif
```

Navigate to wsi-cluster root inside apptainer, then setup **venv**:  

```bash
cd /wsi-cluster

# Create venv
python3 -m venv wsi-cluster
source wsi-cluster/bin/activate

# Install dependencies
pip install -r requirements.txt --no-cache-dir
```

**Venv** setup is now complete, deactivate and exit apptainer.  
```bash
deactivate

exit
```

## Training an SSL Model
All data and outputs should be handled inside the wsi-cluster root but double check the following, edit lightly_scripts/train_ssl.slurm as necessary:
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
Checkpoints are saved to (unless changed): /wsi-cluster/experiments/$METHOD 

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


