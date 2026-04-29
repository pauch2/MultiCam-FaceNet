# test.py (NO detector) - per-identity 80/20 split from a single root
# + threshold sweep + false-positive analysis
import os
import sys
import argparse
from pathlib import Path
import random
import csv

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embedding_model import FaceEmbeddingModel
from database.vector_store import FaceDatabase
import config


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_unique_dir(base_path: Path) -> Path:
    """
    If base_path exists, create base_path_1, base_path_2, ...
    Returns the created directory path.
    """
    if not base_path.exists():
        base_path.mkdir(parents=True)
        return base_path

    counter = 1
    while True:
        new_path = Path(f"{base_path}_{counter}")
        if not new_path.exists():
            new_path.mkdir(parents=True)
            return new_path
        counter += 1


def list_images(root: Path):
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_transform():
    return T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def image_path_to_embedding(model, transform, img_path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor_img = transform(pil_img)
    return model.get_embedding(tensor_img)  # expected np.ndarray


def compute_confusion(y_true, y_pred, class_names):
    idx = {c: i for i, c in enumerate(class_names)}
    C = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        C[idx[t], idx[p]] += 1
    return C


def per_class_metrics(C, class_names):
    metrics = {}
    for i, cls in enumerate(class_names):
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": int(C[i, :].sum())}
    return metrics


def save_confusion_csv(C, class_names, out_csv: Path):
    ensure_dir(out_csv.parent)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(class_names) + "\n")
        for i, cls in enumerate(class_names):
            row = ",".join(str(int(x)) for x in C[i])
            f.write(f"{cls},{row}\n")


def save_confusion_png(C, class_names, out_png: Path, title="Confusion Matrix"):
    ensure_dir(out_png.parent)
    fig = plt.figure(figsize=(max(8, len(class_names) * 0.4), max(6, len(class_names) * 0.35)))
    ax = plt.gca()
    im = ax.imshow(C, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar(im)

    if len(class_names) <= 40:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                v = C[i, j]
                if v > 0:
                    ax.text(j, i, str(int(v)), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_split_manifest(rows, out_csv: Path):
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "split", "path"])
        w.writerows(rows)


# -------------------------
# Threshold sweep + FP analysis
# -------------------------

def parse_thresholds(args):
    """
    args.thresholds can be 'auto' or comma-separated list.
    """
    if args.thresholds != "auto":
        thrs = []
        for x in args.thresholds.split(","):
            x = x.strip()
            if x:
                thrs.append(float(x))
        if not thrs:
            raise ValueError("--thresholds provided but empty after parsing.")
        return thrs

    if args.thr_steps < 2:
        return [float(args.thr_min)]
    step = (args.thr_max - args.thr_min) / (args.thr_steps - 1)
    return [args.thr_min + i * step for i in range(args.thr_steps)]


def analyze_thresholds(raw, thresholds, out_csv_path: Path):
    """
    raw: list of dicts: {true, pred, sim, path}
      - pred: best predicted class (or None)
      - sim: best similarity (or None)
    thresholds: list[float]

    IMPORTANT: assumes higher sim is better; accept if sim >= thr.

    Writes a CSV with:
      threshold,total,accepted,rejected,TP,FP,FN,accept_rate,precision,tpr,fp_rate,fn_rate
    Also returns rows list for plotting.
    """
    rows = []
    for thr in thresholds:
        TP = FP = FN = REJ = 0
        total = 0

        for r in raw:
            true = r["true"]
            pred = r["pred"]
            sim = r["sim"]
            total += 1

            accepted = (pred is not None) and (sim is not None) and (sim >= thr)

            if not accepted:
                REJ += 1
                # "correct reject" approximation: best pred was actually correct but we rejected it
                if pred == true:
                    FN += 1
                continue

            # accepted
            if pred == true:
                TP += 1
            else:
                FP += 1

        accept_rate = (total - REJ) / total if total > 0 else 0.0
        fp_rate = FP / total if total > 0 else 0.0
        fn_rate = FN / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        tpr = TP / total if total > 0 else 0.0

        rows.append({
            "threshold": thr,
            "total": total,
            "accepted": total - REJ,
            "rejected": REJ,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "accept_rate": accept_rate,
            "precision": precision,
            "tpr": tpr,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
        })

    ensure_dir(out_csv_path.parent)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return rows


def plot_threshold_curves(rows, out_png_path: Path):
    th = [r["threshold"] for r in rows]
    fp = [r["fp_rate"] for r in rows]
    fn = [r["fn_rate"] for r in rows]
    prec = [r["precision"] for r in rows]
    accpt = [r["accept_rate"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(th, fp, label="FP rate (wrong accept / total)")
    plt.plot(th, fn, label="FN rate (correct reject / total)")
    plt.plot(th, prec, label="Precision (TP / (TP+FP))")
    plt.plot(th, accpt, label="Accept rate (accepted / total)")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Threshold Sweep Metrics")
    plt.legend()
    plt.tight_layout()

    ensure_dir(out_png_path.parent)
    plt.savefig(out_png_path, dpi=200)
    plt.close()


def save_fp_examples(raw, thresholds, out_dir: Path, top_n: int = 25):
    """
    For each threshold, save a CSV of the highest-similarity false positives:
      true != pred, accepted, sorted by sim desc.

    File: fp_examples_thr_{thr}.csv
    Columns: threshold,true,pred,sim,path
    """
    ensure_dir(out_dir)

    for thr in thresholds:
        fps = []
        for r in raw:
            if r["pred"] is None or r["sim"] is None:
                continue
            if r["sim"] < thr:
                continue
            if r["pred"] != r["true"]:
                fps.append(r)

        fps.sort(key=lambda x: x["sim"], reverse=True)
        fps = fps[:top_n]

        # make filename safe-ish
        thr_str = f"{thr:.4f}".replace(".", "p")
        out_csv = out_dir / f"fp_examples_thr_{thr_str}.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "true", "pred", "sim", "path"])
            for r in fps:
                w.writerow([thr, r["true"], r["pred"], r["sim"], r["path"]])


def main():
    parser = argparse.ArgumentParser(
        description="Build DB + evaluate by splitting each identity folder into 80% train / 20% val (NO detector). "
                    "+ threshold sweep + FP analysis."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root folder containing identity subfolders (your current 'val' root).")

    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction per identity used to build DB (rest used for evaluation).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible splits.")

    parser.add_argument("--max_imgs_per_person", type=int, default=100000,
                        help="Cap number of images per identity before splitting (useful for huge folders).")
    parser.add_argument("--min_imgs_per_person", type=int, default=2,
                        help="Minimum images required to include an identity. Needs >=2 to do a split.")

    # DB query options
    parser.add_argument("--faiss_k", type=int, default=120)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--index_type", type=str, default="hnsw", choices=["hnsw", "flatip", "ivf"])

    # Threshold sweep
    parser.add_argument("--thresholds", type=str, default="auto",
                        help="Comma list like '0.2,0.3,0.4' or 'auto' to sweep.")
    parser.add_argument("--thr_min", type=float, default=0.0)
    parser.add_argument("--thr_max", type=float, default=1.0)
    parser.add_argument("--thr_steps", type=int, default=51)

    # Optional: keep a single-threshold confusion matrix (same as before)
    parser.add_argument("--unknown_threshold", type=float, default=None,
                        help="If set, predictions with similarity < threshold become 'unknown' for confusion matrix.")

    # FP examples
    parser.add_argument("--fp_top_n", type=int, default=25,
                        help="How many top-sim false positives to save per threshold.")

    # DB reuse
    parser.add_argument("--db_path", type=str, default=None,
                        help="Path to the SQLite embedding DB.  Defaults to config.DB_PATH.  "
                             "Set this to point at a previously built DB to skip Stage 1.")
    parser.add_argument("--db_mode", type=str, default=None,
                        choices=["reuse", "new", "overwrite"],
                        help="Non-interactive DB mode.  "
                             "'reuse' = skip Stage 1 if DB exists; "
                             "'new' = always create a fresh DB with a unique name; "
                             "'overwrite' = delete existing DB and rebuild from scratch. "
                             "If not set and a DB already exists, you will be prompted.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_pct = int(round(args.train_ratio * 100))
    val_pct = 100 - train_pct
    base_out_dir = Path(f"runs/eval_{config.MODEL_NAME}_split_{train_pct}_{val_pct}")
    out_dir = make_unique_dir(base_out_dir)
    ensure_dir(out_dir)

    rng = random.Random(args.seed)

    # ── Resolve DB path ────────────────────────────────────────────────────────
    db_path = Path(args.db_path) if args.db_path else Path(config.DB_PATH)

    # ── DB reuse / new / overwrite decision ────────────────────────────────────
    skip_stage1 = False

    if db_path.exists() and db_path.stat().st_size > 0:
        mode = args.db_mode

        if mode is None:
            # Interactive prompt
            print(f"\n  Found existing DB: {db_path}")
            print(f"  Size: {db_path.stat().st_size / 1024:.1f} KB")
            print()
            print("  What do you want to do?")
            print("    [1] Reuse existing DB  — skip Stage 1 (fastest, same split via seed)")
            print("    [2] New DB             — create a fresh DB with a unique auto-name")
            print("    [3] Overwrite          — delete existing DB and rebuild from scratch")
            print()
            while True:
                choice = input("  Enter 1 / 2 / 3: ").strip()
                if choice in ("1", "2", "3"):
                    break
                print("  Please enter 1, 2, or 3.")
            mode = {"1": "reuse", "2": "new", "3": "overwrite"}[choice]

        if mode == "reuse":
            skip_stage1 = True
            print(f"\n  → Reusing existing DB: {db_path}")
            print( "    Stage 1 will be skipped.  Val split is re-derived from the same seed.")

        elif mode == "new":
            # Auto-increment: model_name_1.db, model_name_2.db, ...
            stem   = db_path.stem
            suffix = db_path.suffix
            parent = db_path.parent
            counter = 1
            while True:
                new_db_path = parent / f"{stem}_{counter}{suffix}"
                if not new_db_path.exists():
                    break
                counter += 1
            db_path = new_db_path
            print(f"\n  → Creating new DB: {db_path}")

        elif mode == "overwrite":
            db_path.unlink()
            print(f"\n  → Deleted existing DB.  Rebuilding from scratch: {db_path}")

    elif args.db_mode == "reuse":
        print(f"\n  Warning: --db_mode=reuse specified but no DB found at {db_path}.")
        print( "  Falling back to building a new DB.")

    # ── Init model ────────────────────────────────────────────────────────────
    model = FaceEmbeddingModel(embedding_dim=config.EMBEDDING_DIM)
    if os.path.exists(config.MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location="cpu"))
    else:
        print("Warning: model weights not found; using current weights.")
    model.eval()

    transform = build_transform()
    db = FaceDatabase(db_path=str(db_path))

    # ── Discover classes ──────────────────────────────────────────────────────
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No identity folders found in data_dir={data_dir}")

    kept_classes = []
    split_manifest_rows = []
    val_items = []

    if skip_stage1:
        # ── Stage 1 SKIPPED — still compute the split to know the val set ────
        print("\n[1/3] Stage 1 skipped (reusing existing DB). Computing val split...")
        skipped_too_few = 0

        for cls_dir in class_dirs:
            cls = cls_dir.name
            imgs = list_images(cls_dir)
            if len(imgs) < args.min_imgs_per_person:
                skipped_too_few += 1
                continue
            if len(imgs) > args.max_imgs_per_person:
                imgs = imgs[: args.max_imgs_per_person]

            rng.shuffle(imgs)
            n = len(imgs)
            n_train = int(round(n * args.train_ratio))
            n_train = max(1, min(n - 1, n_train))
            val_imgs = imgs[n_train:]
            if len(val_imgs) == 0:
                continue

            kept_classes.append(cls)
            for vp in val_imgs:
                val_items.append((cls, vp))

        if not kept_classes:
            raise RuntimeError("No classes kept after filtering.")

        print(f"  Classes for evaluation:  {len(kept_classes)}")
        print(f"  Val images:              {len(val_items)}")
        print(f"  Skipped (too few imgs):  {skipped_too_few}")
        manifest_path = out_dir / "split_manifest.csv"  # won't be written in reuse mode

    else:
        # ── Stage 1 NORMAL — build DB from train split ────────────────────────
        train_added = 0
        train_read_fail = 0
        train_emb_fail = 0
        skipped_too_few = 0

        print(f"\n[1/3] Splitting each identity folder and building DB from train split...")
        print(f"  DB path: {db_path}")

        for cls_dir in class_dirs:
            cls = cls_dir.name
            imgs = list_images(cls_dir)
            if len(imgs) < args.min_imgs_per_person:
                skipped_too_few += 1
                continue
            if len(imgs) > args.max_imgs_per_person:
                imgs = imgs[: args.max_imgs_per_person]

            rng.shuffle(imgs)
            n = len(imgs)
            n_train = int(round(n * args.train_ratio))
            n_train = max(1, min(n - 1, n_train))

            train_imgs = imgs[:n_train]
            val_imgs   = imgs[n_train:]
            if len(val_imgs) == 0:
                continue

            kept_classes.append(cls)
            for p in train_imgs:
                split_manifest_rows.append([cls, "train", str(p)])
            for p in val_imgs:
                split_manifest_rows.append([cls, "val", str(p)])

            embs = []
            for img_path in train_imgs:
                emb = image_path_to_embedding(model, transform, img_path)
                if emb is None:
                    train_read_fail += 1
                    continue
                if not isinstance(emb, np.ndarray) or emb.size == 0:
                    train_emb_fail += 1
                    continue
                embs.append(emb)

            if len(embs) == 0:
                print(f"  - {cls}: 0 train embeddings (read/emb failures).")
            else:
                db.add_person_many(cls, embs)
                train_added += len(embs)
                print(f"  - {cls}: train={len(train_imgs)} (added {len(embs)} embs) | val={len(val_imgs)}")

            for vp in val_imgs:
                val_items.append((cls, vp))

        if not kept_classes:
            raise RuntimeError("No classes kept after filtering; check folder structure and min_imgs_per_person.")

        manifest_path = out_dir / "split_manifest.csv"
        write_split_manifest(split_manifest_rows, manifest_path)

        print("\nDB build done.")
        print(f"  DB path:                 {db_path}")
        print(f"  Classes kept:            {len(kept_classes)}")
        print(f"  Total train embeddings:  {train_added}")
        print(f"  Train failed reads:      {train_read_fail}")
        print(f"  Train failed embeddings: {train_emb_fail}")
        print(f"  Skipped (too few imgs):  {skipped_too_few}")
        print(f"  Split manifest:          {manifest_path}")

    # --- Evaluate: collect raw outputs once ---
    print("\n[2/3] Evaluating on per-class val split (collect raw best-match)...")

    raw = []  # list of dicts {true,pred,sim,path}
    val_read_fail = 0
    val_emb_fail = 0

    for true_cls, img_path in val_items:
        emb = image_path_to_embedding(model, transform, img_path)
        if emb is None:
            val_read_fail += 1
            raw.append({"true": true_cls, "pred": None, "sim": None, "path": str(img_path)})
            continue
        if not isinstance(emb, np.ndarray) or emb.size == 0:
            val_emb_fail += 1
            raw.append({"true": true_cls, "pred": None, "sim": None, "path": str(img_path)})
            continue

        name, sim = db.find_closest(
            emb,
            backend="faiss",
            index_type=args.index_type,
            aggregation="topk",
            top_k=args.top_k,
            faiss_k=args.faiss_k,
        )

        raw.append({
            "true": true_cls,
            "pred": name,
            "sim": (None if sim is None else float(sim)),
            "path": str(img_path)
        })

    print(f"  Val failed reads: {val_read_fail}")
    print(f"  Val failed embs:  {val_emb_fail}")
    print(f"  Total val items:  {len(raw)}")

    # --- Threshold sweep + FP analysis ---
    thresholds = parse_thresholds(args)

    thr_csv = out_dir / "threshold_sweep.csv"
    thr_png = out_dir / "threshold_sweep.png"
    rows = analyze_thresholds(raw, thresholds, thr_csv)
    plot_threshold_curves(rows, thr_png)

    fp_dir = out_dir / "fp_examples"
    save_fp_examples(raw, thresholds, fp_dir, top_n=args.fp_top_n)

    # Also print the "best" threshold by precision subject to minimum accept rate (optional heuristic)
    # Here: pick threshold that maximizes precision among thresholds with accept_rate >= 0.5
    best = None
    for r in rows:
        if r["accept_rate"] < 0.5:
            continue
        if best is None or r["precision"] > best["precision"]:
            best = r
    if best is not None:
        print(f"  Heuristic best (accept_rate>=0.5): thr={best['threshold']:.4f}, "
              f"precision={best['precision']:.4f}, fp_rate={best['fp_rate']:.4f}, "
              f"accept_rate={best['accept_rate']:.4f}")

    # --- Optional: single-threshold confusion matrix (like your previous output) ---
    # If unknown_threshold is not provided, we will still build confusion WITHOUT "unknown" by always accepting pred
    print("\n[3/3] Saving confusion matrix (optional single threshold)...")

    eval_class_names = sorted(kept_classes)
    if args.unknown_threshold is not None and "unknown" not in eval_class_names:
        eval_class_names.append("unknown")

    y_true, y_pred = [], []
    for r in raw:
        true = r["true"]
        pred = r["pred"] if r["pred"] is not None else "unknown"
        sim = r["sim"]

        if args.unknown_threshold is not None:
            if sim is None or (pred == "unknown") or (float(sim) < float(args.unknown_threshold)):
                pred = "unknown"

        y_true.append(true)
        y_pred.append(pred)

    C = compute_confusion(y_true, y_pred, eval_class_names)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    acc = correct / total if total > 0 else 0.0
    metrics = per_class_metrics(C, eval_class_names)

    out_cm_csv = out_dir / "confusion_matrix.csv"
    out_cm_png = out_dir / "confusion_matrix.png"
    out_txt = out_dir / "report.txt"

    save_confusion_csv(C, eval_class_names, out_cm_csv)

    title = f"Confusion Matrix ({train_pct}/{val_pct} split, NO detector)"
    if args.unknown_threshold is not None:
        title += f" | thr={args.unknown_threshold:.4f}"
    save_confusion_png(C, eval_class_names, out_cm_png, title=title)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"EVALUATION REPORT (NO detector, per-class ({train_pct}/{val_pct} split))\n")
        f.write("======================================================\n\n")
        f.write(f"Data dir: {data_dir}\n")
        f.write(f"DB path:  {db_path}\n")
        f.write(f"DB mode:  {'reuse (Stage 1 skipped)' if skip_stage1 else 'built from scratch'}\n")
        f.write(f"Out dir:  {out_dir}\n")
        f.write(f"Classes kept: {len(kept_classes)}\n")
        f.write(f"Train ratio: {args.train_ratio}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Index type: {args.index_type}\n")
        f.write(f"top_k: {args.top_k}, faiss_k: {args.faiss_k}\n")
        f.write(f"Val failed reads: {val_read_fail}\n")
        f.write(f"Val failed embs:  {val_emb_fail}\n\n")

        f.write("THRESHOLD SWEEP\n")
        f.write("--------------\n")
        f.write(f"thresholds: {args.thresholds}\n")
        if args.thresholds == "auto":
            f.write(f"thr_min={args.thr_min}, thr_max={args.thr_max}, thr_steps={args.thr_steps}\n")
        f.write(f"sweep_csv: {thr_csv}\n")
        f.write(f"sweep_png: {thr_png}\n")
        f.write(f"fp_examples_dir: {fp_dir}\n\n")

        f.write("SINGLE-THRESHOLD CONFUSION (optional)\n")
        f.write("------------------------------------\n")
        f.write(f"unknown_threshold: {args.unknown_threshold}\n")
        f.write(f"Total val samples: {total}\n")
        f.write(f"Accuracy (Top-1):  {acc:.4f}\n\n")

        f.write("PER-CLASS METRICS (from confusion)\n")
        f.write("---------------------------------\n")
        f.write("class,precision,recall,f1,support\n")
        for cls in eval_class_names:
            m = metrics[cls]
            f.write(f"{cls},{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f},{m['support']}\n")

    print("Saved artifacts:")
    print(f"  - {manifest_path}")
    print(f"  - {thr_csv}")
    print(f"  - {thr_png}")
    print(f"  - {fp_dir}/fp_examples_thr_*.csv")
    print(f"  - {out_cm_csv}")
    print(f"  - {out_cm_png}")
    print(f"  - {out_txt}")


if __name__ == "__main__":
    main()