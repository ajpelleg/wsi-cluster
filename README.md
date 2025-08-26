# wsi-cluster

A reproducible pipeline for **self-supervised learning (SSL) feature embeddings** on whole slide image (WSI) patches.  
This repo trains SSL models (SimCLR, MoCo, DINO, DenseCL) using PyTorch Lightning + Lightly, then extracts embeddings for downstream clustering or visualization.

---
## Environment Setup

This repo uses a **Python venv**.  

```bash
# Create a virtual environment (while in wsi-cluster root)
python3 -m venv wsi-cluster
source wsi-cluster/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (choose CUDA version)
pip install --index-url https://download.pytorch.org/whl/cu118 -r requirements.txt   # for CUDA 11.8
# pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements.txt # for CUDA 12.1
# pip install --index-url https://download.pytorch.org/whl/cpu   -r requirements.txt # CPU only
```

## Training an SSL Model
Be sure to check the following:
  * Paths to data
  * Output directory paths
  * Any other paths in train_ssl.slurm
Checkpoints are saved to: <output_dir>/<run_name>/
Submit the job with the following (from the wsi-cluster root):
```bash
sbatch lightly_scripts/train_ssl.slurm
```
