#!/usr/bin/env python
"""
extract_features.py

Load a trained Lightly SSL checkpoint, rebuild the exact same model,
run the backbone on raw images, and save per-image feature tensors to .pt files.
Produces both a master .pt with all features and individual .pt files per image.
"""
import os, argparse, math, json
import torch
from PIL import Image
from torchvision import transforms, models as tv_models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform   import MoCoV2Transform
from lightly.transforms         import DenseCLTransform
from lightly.loss               import NTXentLoss, DINOLoss
from lightly.models.modules.heads import (
    SimCLRProjectionHead,
    MoCoProjectionHead,
    DINOProjectionHead,
    DenseCLProjectionHead,
)

from train_ssl import SSLModel
# 1) Simple inference dataset for single-view loading
class InferenceDataset(Dataset):
    def __init__(self, file_paths, preprocess):
        self.file_paths = file_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img  = Image.open(path).convert("RGB")
        return self.preprocess(img), os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract backbone features from a Lightly SSL checkpoint"
    )
    parser.add_argument("ckpt",           help="Path to the .ckpt file to load")
    parser.add_argument("--data_dir",     required=True, help="Root data directory")
    parser.add_argument("--image_folder", required=True, help="Subfolder under data_dir")
    parser.add_argument("--method",       required=True,
                        choices=["simclr","moco","dino","densecl"])
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=8)
    parser.add_argument("--output_root",  default="/PathLDM/experiments/features")
    args = parser.parse_args()

    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Rebuild model & load weights
    # 2a) Load the saved config.json from the same folder as your .ckpt
    ckpt_dir = os.path.dirname(args.ckpt)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # 2b) Instantiate the same backbone
    resnet = getattr(tv_models, cfg["backbone"])(pretrained=True)
    if cfg["method"] == "densecl":
        # DenseCL uses only the conv trunk
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        feat_dim  = resnet.fc.in_features
    else:
        # simclr, moco, dino drop the final fc
        backbone = resnet
        feat_dim  = backbone.fc.in_features
        backbone.fc = nn.Identity()

    # 2c) Build the same head & criterion
    method = cfg["method"]
    if method == "simclr":
        head      = SimCLRProjectionHead(feat_dim, hidden_dim=512, output_dim=256)
        criterion = NTXentLoss(temperature=0.5)
    elif method == "moco":
        head      = MoCoProjectionHead(feat_dim, hidden_dim=2048, output_dim=256)
        criterion = NTXentLoss(temperature=0.2)
    elif method == "dino":
        head      = DINOProjectionHead(
            input_dim=feat_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            output_dim=256,
            batch_norm=True
        )
        criterion = DINOLoss(output_dim=256)
    elif method == "densecl":
        head = (
            DenseCLProjectionHead(input_dim=feat_dim, hidden_dim=feat_dim, output_dim=256),
            DenseCLProjectionHead(input_dim=feat_dim, hidden_dim=feat_dim, output_dim=256)
        )
        criterion = (
            NTXentLoss(temperature=0.2),
            NTXentLoss(temperature=0.2)
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    # 2d) Import and instantiate your LightningModule
    
    model = SSLModel(
        backbone       = backbone,
        head           = head,
        criterion      = criterion,
        lr             = cfg["lr"],
        weight_decay   = cfg["weight_decay"],
        method         = method
    )

    # 2e) Load weights from checkpoint
    ckpt_data = torch.load(args.ckpt, map_location="cpu")
    state     = ckpt_data["state_dict"]
    model.load_state_dict(state)
    model.eval().cuda()

    # ─────────────────────────────────────────────────────────────────────────────

    if method == "densecl":
        def extract(x):
            _, g, _ = model(x)
            return g
    elif method == "dino":
        def extract(x):
            # student path already has avgpool+head
            return model.student_head(
                model.student_backbone(x).flatten(start_dim=1)
            )
    else:
        # simclr & moco: backbone→flatten→projection_head
        def extract(x):
            return model.projection_head(
                model.backbone(x).flatten(start_dim=1)
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Build standard torchvision preprocessing from default_cfg

    cfg_back = getattr(model.backbone, "default_cfg", {}) or {}
    H, W        = cfg_back.get("input_size", (3,224,224))[1:]
    crop_pct    = cfg_back.get("crop_pct", 0.875)
    resize_short= int(math.floor(H / crop_pct))
    mean        = cfg_back.get("mean", [0.485,0.456,0.406])
    std         = cfg_back.get("std",  [0.229,0.224,0.225])

    preprocess = transforms.Compose([
        transforms.Resize(resize_short),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ─────────────────────────────────────────────────────────────────────────────
    # 5) Gather all image files & DataLoader

    image_dir = os.path.join(args.data_dir, args.image_folder)
    file_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png",".jpg",".jpeg"))
    ])
    dataset = InferenceDataset(file_paths, preprocess)
    loader  = DataLoader(
        dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # 6) Prepare output dirs & run inference

    master_dir = os.path.join(args.output_root, method)
    indiv_dir  = os.path.join(master_dir, "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    feats_dict = {}
    with torch.no_grad():
        for x, names in loader:
            x     = x.cuda(non_blocking=True)
            feats = extract(x).cpu()
            for n, f in zip(names, feats):
                feats_dict[n] = f
                torch.save(f, os.path.join(indiv_dir, f"{n}.pt"))

    # ─────────────────────────────────────────────────────────────────────────────
    # 7) Save the master feature‐dict

    out_master = os.path.join(master_dir, f"{method}_features.pt")
    torch.save(feats_dict, out_master)
    print(f"Saved {len(feats_dict)} patches → {out_master}")


if __name__ == "__main__":
    main()
