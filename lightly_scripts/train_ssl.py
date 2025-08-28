#!/usr/bin/env python
"""
train_ssl.py

Self-supervised training with Lightly + PyTorch Lightning.
Supports: simclr, moco, dino, densecl.
"""
import os, re, argparse, json
from PIL import Image
from datetime import datetime
import copy

import torch, pytorch_lightning as pl
import torchvision.models as tv_models
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform   import MoCoV2Transform
from lightly.transforms         import DenseCLTransform
from lightly.data import LightlyDataset
from lightly.transforms import DINOTransform
from lightly.loss import NTXentLoss, DINOLoss
from lightly.models             import utils as lightly_utils
from lightly.utils.scheduler    import cosine_schedule
from lightly.models.modules.heads import (
    SimCLRProjectionHead,
    MoCoProjectionHead,
    DINOProjectionHead,
    DenseCLProjectionHead,
)
from lightly.data.collate import DINOCollateFunction
# ─────────────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform  = transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img  = Image.open(path).convert("RGB")
        views = self.transform(img)
        v0, v1 = views[0], views[1]
        label    = 0
        filename = os.path.basename(path)
        return (v0, v1), label, filename
        
        

# ─────────────────────────────────────────────────────────────────────────────
class MetricsLogger(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {"train_loss": [], "val_loss": []}

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.history["train_loss"].append(loss.item())
            with open(f"{self.output_dir}/metrics.json", "w") as f:
                json.dump(self.history, f, indent=4)

    def on_validation_epoch_end(self, trainer, pl_module, outputs=None):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.history["val_loss"].append(loss.item())
            with open(f"{self.output_dir}/metrics.json", "w") as f:
                json.dump(self.history, f, indent=4)

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised Lightly training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir",    default="features/data",
                        help="Up-level data directory")
    parser.add_argument("--image_folder", default="tumor_segmentation_v2_05mpp_256/tiles/images")
    parser.add_argument("--output_dir",  default="features/checkpoints")
    parser.add_argument("--method",      choices=["simclr","moco","dino","densecl"], required=True)
    parser.add_argument("--backbone",    choices=["resnet18","resnet50"], default="resnet50")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--weight_decay",type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--gpus",        type=int, default=2)
    parser.add_argument("--save_top_k",  type=int, default=1)
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--run_name",    type=str, default=None)
    parser.add_argument("--occ_split", action="store_true", help="Use OCC-specific splitting. If not set, perform a regular random split")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
def train_val_split(file_paths, val_frac):
    fnames = [os.path.basename(p) for p in file_paths]
    groups = [re.split(r"_x_", fn)[0] for fn in fnames]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(splitter.split(file_paths, groups=groups))
    return [file_paths[i] for i in train_idx], [file_paths[i] for i in val_idx]
    
def regular_split(file_paths, val_frac):
    train_paths, val_paths = train_test_split(
        file_paths,
        test_size=val_frac,
        random_state=42
    )
    return train_paths, val_paths
    
# ─────────────────────────────────────────────────────────────────────────────
class SSLModel(pl.LightningModule):
    def __init__(self, backbone, head, criterion, lr, weight_decay, method: str):
        super().__init__()
        self.method = method
        self.save_hyperparameters("lr", "weight_decay", "method")
        self.backbone = backbone

        if method == "densecl":
            # head is a tuple (head_g, head_l), criterion is (crit_g, crit_l)
            self.projection_head_global, self.projection_head_local = head
            self.criterion_global,     self.criterion_local = criterion
            
            self.crit_g = self.criterion_global
            self.crit_l = self.criterion_local
            # create a momentum encoder copy
            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_global_momentum = copy.deepcopy(self.projection_head_global)
            self.projection_head_local_momentum  = copy.deepcopy(self.projection_head_local)

            # freeze gradients on all momentum weights
            lightly_utils.deactivate_requires_grad(self.backbone_momentum)
            lightly_utils.deactivate_requires_grad(self.projection_head_global_momentum)
            lightly_utils.deactivate_requires_grad(self.projection_head_local_momentum)
            
            # Aliases for validation_step / training_step
            self.backbone_m = self.backbone_momentum
            self.head_g_m   = self.projection_head_global_momentum
            self.head_l_m   = self.projection_head_local_momentum

            # spatial pooling for global representations
            self.pool = nn.AdaptiveAvgPool2d((1,1))
        elif method == "dino":
            # 1) student network & head
            self.student_backbone = backbone
            self.student_head     = head

            for m in self.student_head.modules():
                try:
                    nn_utils.remove_weight_norm(m)
                except (ValueError, AttributeError):
                # skip modules that don’t have weight_norm
                    pass

            # 2) teacher network & head (deepcopy + freeze)
            self.teacher_backbone = copy.deepcopy(self.student_backbone)
            self.teacher_head     = copy.deepcopy(self.student_head)
            lightly_utils.deactivate_requires_grad(self.teacher_backbone)
            lightly_utils.deactivate_requires_grad(self.teacher_head)

            # 3) DINOLoss (with your chosen dims + teacher‐temp warmup)
            self.criterion = DINOLoss(
                output_dim   = 256,
                warmup_teacher_temp_epochs = 5
            )

            # 4) helper pool if your backbone doesn’t already include avgpool
            self.pool = nn.AdaptiveAvgPool2d((1,1))

            # 5) expose them for training_step
            self.backbone        = self.student_backbone
            self.projection_head = self.student_head
            
        elif method == "moco":
            # student network + head
            self.backbone        = backbone
            self.projection_head = head
            self.criterion       = criterion

            # momentum (key) encoder copies
            self.backbone_m = copy.deepcopy(self.backbone)
            self.projection_head_m = copy.deepcopy(self.projection_head)
            lightly_utils.deactivate_requires_grad(self.backbone_m)
            lightly_utils.deactivate_requires_grad(self.projection_head_m)
            
        elif method == "simclr":
            # 1) standard backbone + SimCLR head
            self.backbone        = backbone
            self.projection_head = head
            self.criterion       = criterion

        else:
            raise ValueError(f"Unknown method {method}")
            
    def forward(self, x):
        if self.method == "densecl":
            # query path
            query_features = self.backbone(x)                                 # B×C×H×W
            query_global   = self.pool(query_features).flatten(start_dim=1)   # B×C
            query_global   = self.projection_head_global(query_global)        # B×D
            query_local    = query_features.flatten(start_dim=2).permute(0,2,1)  # B×(H*W)×C
            query_local    = self.projection_head_local(query_local)          # B×(H*W)×D
            return query_features, query_global, query_local
        elif self.method == "dino":
            y = self.student_backbone(x).flatten(start_dim=1)
            return self.student_head(y)
        elif self.method == "simclr":
            # exactly SimCLR: backbone→flatten→projection_head
            feats = self.backbone(x).flatten(start_dim=1)
            return self.projection_head(feats) 
        elif self.method == "moco":
            feats = self.backbone(x).flatten(start_dim=1)
            return self.projection_head(feats)
            
    @torch.no_grad()
    def forward_teacher(self, x): #for dino
        y = self.teacher_backbone(x).flatten(start_dim=1)
        return self.teacher_head(y)
        
    @torch.no_grad()
    def forward_momentum(self, x): #for densecl
        if self.method == "densecl":
            key_features = self.backbone_momentum(x)
            key_global   = self.pool(key_features).flatten(start_dim=1)
            key_global   = self.projection_head_global_momentum(key_global)
            key_local    = key_features.flatten(start_dim=2).permute(0,2,1)
            key_local    = self.projection_head_local_momentum(key_local)
            return key_features, key_global, key_local
        elif self.method == "moco":
            feats = self.backbone_m(x).flatten(start_dim=1)
            return self.projection_head_m(feats)
    def training_step(self, batch, batch_idx):
        if self.method == "densecl":
            # batch yields ((x_q, x_k), _, _)
            (x_q, x_k), _, _ = batch
            # 1) update momentum encoders
            m = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1.0)
            lightly_utils.update_momentum(self.backbone,
                                          self.backbone_momentum,
                                          m=m)
            lightly_utils.update_momentum(self.projection_head_global,
                                          self.projection_head_global_momentum,
                                          m=m)
            lightly_utils.update_momentum(self.projection_head_local,
                                          self.projection_head_local_momentum,
                                          m=m)

            # 2) forward query & key
            q_feats, q_global, q_local = self(x_q)
            k_feats, k_global, k_local = self.forward_momentum(x_k)

            
            # 1) flatten the raw conv outputs into (B, H*W, C)
            q_feats_flat = q_feats.flatten(start_dim=2).permute(0, 2, 1)
            k_feats_flat = k_feats.flatten(start_dim=2).permute(0, 2, 1)
            
            # 2) then pick correspondences on these 3-D feature maps
            k_local = lightly_utils.select_most_similar(q_feats_flat, k_feats_flat, k_local)

            # 4) flatten local dims for NTXent
            q_local_flat = q_local.flatten(end_dim=1)
            k_local_flat = k_local.flatten(end_dim=1)

            # 5) global + local losses
            loss_g = self.criterion_global(q_global, k_global)
            loss_l = self.criterion_local(q_local_flat, k_local_flat)
            λ      = 0.5
            loss   = (1 - λ) * loss_g + λ * loss_l
            batch_size = x_q.size(0)
            
        elif self.method == "dino":
            views, _, _ = batch

            # a) update momentum
            m = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1.0)
            lightly_utils.update_momentum(
              self.student_backbone, self.teacher_backbone, m=m)
            lightly_utils.update_momentum(
              self.student_head,     self.teacher_head,     m=m)

            # b) student & teacher forward
            student_out = [ self(v)            for v in views ]
            teacher_out = [ self.forward_teacher(v) for v in views[:2] ]

            # c) compute loss & cancel last‐layer gradients
            loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
            # freeze the last layer of the student head as per the paper
            self.student_head.cancel_last_layer_gradients(self.current_epoch)

            batch_size = views[0].size(0)
            
        elif self.method == "moco":
            # batch: ((x_q, x_k), _, _)
            (x0, x1), _, _ = batch

            # 1) momentum‐update on both backbone & head
            m = cosine_schedule(self.current_epoch, self.trainer.max_epochs, 0.996, 1.0)
            lightly_utils.update_momentum(self.backbone,        self.backbone_m,        m=m)
            lightly_utils.update_momentum(self.projection_head, self.projection_head_m, m=m)

            # 2) forward query vs. key
            q = self.backbone(x0).flatten(1)
            q = self.projection_head(q)
            k = self.forward_momentum(x1)

            # 3) contrastive loss
            loss = self.criterion(q, k)
            batch_size = x0.size(0)
        elif self.method == "simclr":
            # batch: ((x0, x1), _, _)
            (x0, x1), _, _ = batch
            # 1) two views through your model()
            z0 = self(x0)
            z1 = self(x1)
            # 2) contrastive loss
            loss = self.criterion(z0, z1)
            batch_size = x0.size(0)
        else:
            raise ValueError(f"Unknown method {self.method}")
            
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.method == "densecl":
            # for now, just do a quick global‐only val
            (x_q, x_k), _, _ = batch
            _, gq, _ = self(x_q)
            with torch.no_grad():
                feats_k = self.backbone_m(x_k)
                gk = self.pool(feats_k).flatten(1)
                gk = self.head_g_m(gk)
            loss = self.crit_g(gq, gk)
            batch_size = x_q.size(0)
        elif self.method == "moco":
            (x0, x1), _, _ = batch
            q = self(x0)  
            k = self.forward_momentum(x1)
            loss = self.criterion(q, k)
            batch_size = x0.size(0)
        elif self.method == "dino":
            views, _, _ = batch
            stu = [ self(v) for v in views ]
            tea = stu[:2]
            loss = self.criterion(tea, stu)
            batch_size = views[0].size(0)
        elif self.method == "simclr":
            # SimCLR validation: same as training minus backward
            (x0, x1), _, _ = batch
            z0 = self(x0)
            z1 = self(x1)
            loss = self.criterion(z0, z1)
            batch_size = x0.size(0)
        else:
            raise ValueError(f"Unknown method {self.method}")
            
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        return

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9
        )

# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ① Seed
    if args.seed is None:
        args.seed = int(torch.randint(0,10000,(1,)).item())
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    # ② Run name + output folder
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + args.method
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ③ Dump CLI → config.json
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # 4.1 discover files
    image_dir = os.path.join(args.data_dir, args.image_folder)
    file_paths = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png",".jpg",".jpeg"))
    )
    
    if args.occ_split:
        print("Using OCC split...")
        train_files, val_files = train_val_split(file_paths, args.val_split)
    else:
        print("Using regular random split...")
        train_files, val_files = regular_split(file_paths, args.val_split)
    
    #train_files, val_files = train_val_split(file_paths, args.val_split)

    # 4.3 backbone → transform → head → criterion
    resnet = getattr(tv_models, args.backbone)(pretrained=True)

    # ResNet.fc.in_features is the channel count *before* avgpool
    conv_feat_dim = resnet.fc.in_features
    
    if args.method == "densecl":
        # Keep only the conv trunk: drop avgpool + fc
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        feat_dim  = conv_feat_dim
    else:
        # Keep the whole model minus its final fc
        backbone = resnet
        feat_dim  = backbone.fc.in_features
        backbone.fc = nn.Identity()

#output/embedding dim is 256 for all models!!

    if args.method=="simclr":
        transform = SimCLRTransform(input_size=256, cj_prob=0.5)
        head = SimCLRProjectionHead(feat_dim, hidden_dim=512, output_dim=256)
        criterion = NTXentLoss(temperature=0.5)
    elif args.method=="moco":
        transform = MoCoV2Transform(input_size=256)
        head = MoCoProjectionHead(feat_dim, hidden_dim=2048, output_dim=256)
        criterion = NTXentLoss(temperature=0.2)
    elif args.method=="dino":
        head      = DINOProjectionHead(
            input_dim=feat_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            output_dim=256,
            batch_norm=True,
            freeze_last_layer=0
        )
        criterion = DINOLoss(output_dim=256)
    elif args.method=="densecl":
        # 1) DenseCL dataset‐level transform
        transform = DenseCLTransform(input_size=256)

        # 2) Two projection heads: global and local
        head_g = DenseCLProjectionHead(
            input_dim=feat_dim,
            hidden_dim=feat_dim,
            output_dim=256
        )
        head_l = DenseCLProjectionHead(
            input_dim=feat_dim,
            hidden_dim=feat_dim,
            output_dim=256
        )
        head = (head_g, head_l)

        # 3) Two NTXent losses (global + local)
        criterion = (
            NTXentLoss(temperature=0.2), 
            NTXentLoss(temperature=0.2)
        )
    else:
        raise ValueError(f"Unknown method {args.method}")
        
    # 5) build datasets + loaders
    if args.method == "densecl":
        # PatchDataset will feed transform(img)→(x_q, x_k)
        train_ds = PatchDataset(train_files, transform)
        val_ds   = PatchDataset(val_files,   transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader   = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    
    elif args.method == "dino":
        # 2) LightlyDataset wraps your file list + transform
        train_ds = LightlyDataset(
            input_dir=image_dir,
            transform=None,
            filenames=[os.path.basename(p) for p in train_files]
        )
        val_ds = LightlyDataset(
            input_dir=image_dir,
            transform=None,
            filenames=[os.path.basename(p) for p in val_files]
        )

           # All of the multi‑crop logic lives in the collate function now:
        collate_fn = DINOCollateFunction(
            global_crop_size=256,
            local_crop_size=128,
            global_crop_scale=(0.4, 1.0),
            local_crop_scale=(0.05, 0.4),
            n_local_views=6,
            hf_prob=0.5,
            vf_prob=0.0,
            rr_prob=0.0,
            cj_prob=0.8,
            solarization_prob=0.2
        )

        # 4) DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        train_ds = PatchDataset(train_files, transform)
        val_ds   = PatchDataset(val_files,   transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader   = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )


    # 4.7 callbacks + Trainer
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=args.save_top_k,
        mode="min",
        dirpath=args.output_dir,
        filename=f"{args.method}-{now}" + "-{epoch:02d}-{val_loss:.4f}"
    )

    metrics_cb = MetricsLogger(args.output_dir)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,               
        sync_batchnorm=True,
        callbacks=[ckpt_cb, metrics_cb],
        default_root_dir=args.output_dir,
    )

    # 4.8 fit + 4.9 final checkpoint
    trainer.fit(SSLModel(backbone, head, criterion, args.lr, args.weight_decay, args.method),
                train_loader, val_loader)

    final_ckpt = os.path.join(args.output_dir,
                              f"{args.method}_final_{now}.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"Done → all artifacts are in {args.output_dir}")

if __name__ == "__main__":
    main()