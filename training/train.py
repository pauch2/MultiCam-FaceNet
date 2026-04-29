"""
train.py
--------
Training script for FaceEmbeddingModel.

Supports two loss functions selectable via --loss:
  triplet   Semi-Hard Triplet Loss  (original, no class labels needed)
  arcface   ArcFace Margin Loss     (needs class labels; generally trains faster
                                     and achieves tighter class boundaries)

Quick start
-----------
  python train.py /path/to/dataset                    # triplet (default)
  python train.py /path/to/dataset --loss arcface     # ArcFace
  python train.py /path/to/dataset --loss arcface --arc_margin 0.5 --arc_scale 64
  python train.py /path/to/dataset --loss arcface --resume

Checkpoints
-----------
  Backbone saved to config.MODEL_WEIGHTS_PATH.
  ArcFace head saved separately as <MODEL_WEIGHTS_PATH>.archead.pth
  (only needed to resume training, never for inference).
"""

import os
import sys
import argparse

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import TripletFaceDataset
from arcface_dataset import LabeledFaceDataset
from triplet_loss import SemiHardTripletLoss
from arcface_loss import ArcFaceHead
from models.embedding_model import FaceEmbeddingModel
import config


def _arc_head_path() -> str:
    return config.MODEL_WEIGHTS_PATH + ".archead.pth"


def _make_optimizer(model, head, lr):
    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())
    return optim.Adam(params, lr=lr)


# ── Triplet training loop ─────────────────────────────────────────────────────

def _train_triplet(args, device):
    print("Loss: Semi-Hard Triplet Loss")

    dataset    = TripletFaceDataset(args.dataset_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=min(8, os.cpu_count() or 1),
    )

    model     = FaceEmbeddingModel(embedding_dim=config.EMBEDDING_DIM).to(device)
    criterion = SemiHardTripletLoss(margin=config.MARGIN)
    optimizer = _make_optimizer(model, None, args.lr)
    best_loss = float("inf")

    if args.resume and os.path.exists(config.MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=device))
        print(f"Resumed from {config.MODEL_WEIGHTS_PATH}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, (anchor, positive, negative) in pbar:
            anchor   = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()

            loss_val      = float(loss.item())
            running_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             avg=f"{running_loss/(batch_idx+1):.4f}")

        avg_loss = running_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_WEIGHTS_PATH)
            print(f"--> Saved best model (loss {best_loss:.4f})")


# ── ArcFace training loop ─────────────────────────────────────────────────────

def _train_arcface(args, device):
    dataset = LabeledFaceDataset(args.dataset_dir)
    if dataset.num_classes < 2:
        raise RuntimeError("ArcFace requires at least 2 identity classes.")

    C = dataset.num_classes

    # ── Auto-tune margin and scale if not explicitly set ───────────────────
    # margin: tighter for large datasets; looser for small ones
    # scale:  rule of thumb s ≈ log(C-1)*8, capped at 64
    if args.arc_margin is None:
        if C < 200:   args.arc_margin = 0.2
        elif C < 500: args.arc_margin = 0.3
        elif C < 2000: args.arc_margin = 0.4
        else:          args.arc_margin = 0.5
        print(f"  arc_margin auto-set to {args.arc_margin} for {C} classes")

    if args.arc_scale is None:
        import math as _math
        args.arc_scale = float(min(64, max(16, round(_math.log(max(C-1, 1)) * 8))))
        print(f"  arc_scale  auto-set to {args.arc_scale} for {C} classes")

    print(f"Loss: ArcFace  margin={args.arc_margin}  scale={args.arc_scale}  "
          f"easy_margin={args.arc_easy_margin}  warmup={args.arc_warmup}")
    print(f"  num_classes = {C}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=min(8, os.cpu_count() or 1),
        drop_last=True,
    )

    model = FaceEmbeddingModel(embedding_dim=config.EMBEDDING_DIM).to(device)

    # Detect the actual embedding dimension by running a dummy forward pass.
    # This handles the case where config.EMBEDDING_DIM doesn't match what the
    # backbone actually outputs (e.g. swin_t outputs 768 but config says 128).
    model.eval()
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(device)
        _actual_dim = model(_dummy).shape[1]
    model.train()

    if _actual_dim != config.EMBEDDING_DIM:
        print(f"  Note: backbone outputs {_actual_dim}-dim embeddings "
              f"(config.EMBEDDING_DIM={config.EMBEDDING_DIM}). "
              f"Using {_actual_dim} for ArcFace head.")

    head  = ArcFaceHead(
        embedding_dim=_actual_dim,
        num_classes=dataset.num_classes,
        margin=args.arc_margin,
        scale=args.arc_scale,
        easy_margin=args.arc_easy_margin,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, head, args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    best_loss = float("inf")

    if args.resume:
        if os.path.exists(config.MODEL_WEIGHTS_PATH):
            model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=device))
            print(f"Resumed backbone from {config.MODEL_WEIGHTS_PATH}")
        if os.path.exists(_arc_head_path()):
            head.load_state_dict(torch.load(_arc_head_path(), map_location=device))
            print(f"Resumed ArcFace head from {_arc_head_path()}")

    for epoch in range(args.epochs):
        # Warmup: gradually ramp margin from 0 → target over arc_warmup epochs.
        # At init cos(θ)≈0, so adding full margin immediately makes true-class
        # logit negative → network never learns. Ramp gives it time to organise.
        if args.arc_warmup > 0 and epoch < args.arc_warmup:
            ramp = (epoch + 1) / args.arc_warmup           # 0 → 1
            head.margin = args.arc_margin * ramp
            head.cos_m  = math.cos(head.margin)
            head.sin_m  = math.sin(head.margin)
            head.th     = math.cos(math.pi - head.margin)
            head.mm     = math.sin(math.pi - head.margin) * head.margin
        elif epoch == args.arc_warmup:
            head.margin = args.arc_margin                   # full margin
            head.cos_m  = math.cos(head.margin)
            head.sin_m  = math.sin(head.margin)
            head.th     = math.cos(math.pi - head.margin)
            head.mm     = math.sin(math.pi - head.margin) * head.margin

        model.train(); head.train()
        running_loss = 0.0
        correct = total = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, (imgs, labels) in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            emb    = model(imgs)          # (B, D) — L2 normalised by backbone
            logits = head(emb, labels)    # (B, C) — scaled cosine + margin
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping avoids early-epoch spikes common with ArcFace
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=5.0
            )
            optimizer.step()

            loss_val      = float(loss.item())
            running_loss += loss_val
            with torch.no_grad():
                correct += (logits.argmax(1) == labels).sum().item()
                total   += labels.size(0)

            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             avg=f"{running_loss/(batch_idx+1):.4f}",
                             acc=f"{100*correct/max(1,total):.1f}%")

        scheduler.step()
        avg_loss = running_loss / max(1, len(dataloader))
        acc      = 100.0 * correct / max(1, total)
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {avg_loss:.4f}  Acc: {acc:.1f}%  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_WEIGHTS_PATH)
            torch.save(head.state_dict(), _arc_head_path())
            print(f"--> Saved best model (loss {best_loss:.4f})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train FaceEmbeddingModel with Triplet or ArcFace loss."
    )
    parser.add_argument("dataset_dir",       type=str)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=None,
                        help="Learning rate. Defaults to 1e-3 for triplet, 1e-4 for arcface.")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--loss",            type=str,   default="triplet",
                        choices=["triplet", "arcface"])
    parser.add_argument("--arc_margin",      type=float, default=None,
                        help="ArcFace angular margin (radians). "
                             "Auto-set based on num_classes if not specified.")
    parser.add_argument("--arc_scale",       type=float, default=None,
                        help="ArcFace feature scale s. "
                             "Auto-set based on num_classes if not specified.")
    parser.add_argument("--arc_easy_margin", action="store_true", default=True,
                        help="Use easy-margin variant. ON by default — "
                             "prevents 0%% accuracy at init. Use --no_arc_easy_margin to disable.")
    parser.add_argument("--no_arc_easy_margin", dest="arc_easy_margin", action="store_false")
    parser.add_argument("--arc_warmup",      type=int, default=5,
                        help="Epochs to train with margin=0 before applying full margin (default 5). "
                             "Helps escape the zero-accuracy trap on difficult datasets.")
    parser.add_argument("--gpu",             type=int,   default=None,
                        help="GPU index. Falls back to TRAIN_GPU_ID env var then auto.")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        gpu_id = args.gpu if args.gpu is not None else int(os.environ.get("TRAIN_GPU_ID", "0"))
        gpu_id = min(gpu_id, torch.cuda.device_count() - 1)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Output: {config.MODEL_WEIGHTS_PATH}")

    # Default LR depends on loss — ArcFace needs a lower rate
    if args.lr is None:
        args.lr = 1e-4 if args.loss == "arcface" else 1e-3
    print(f"Learning rate: {args.lr}")

    if args.loss == "arcface":
        _train_arcface(args, device)
    else:
        _train_triplet(args, device)


if __name__ == "__main__":
    main()