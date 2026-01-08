"""
Compare ResNet-50 (timm) vs ViT-Base (HuggingFace) on ImageNet validation subset.

This script evaluates both models on the same images and reports:
- Top-1 and Top-5 accuracy for each model
- Per-image predictions (where they agree/disagree)
- Inference latency comparison
- Cases where one model is correct and the other is wrong

Usage:
    python compare_models.py
    python compare_models.py --val-dir imagenet-val-short --batch-size 1
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
import timm
from transformers import ViTImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
import numpy as np


def load_id2label(json_path: str) -> dict:
    """Load ImageNet id2label mapping from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_wnid_to_idx(id2label: dict) -> dict:
    """
    Build a mapping from WordNet ID (e.g., 'n01440764') to ImageNet class index.
    This requires the label2id mapping which we can infer from id2label.
    """
    # We need to map folder names (wnid) to class indices
    # The standard ImageNet ordering is alphabetical by wnid
    # We'll load the mapping from a known source
    wnid_to_idx = {}
    # Load from HuggingFace's mapping
    try:
        from urllib.request import urlopen
        with urlopen(
            "https://huggingface.co/datasets/huggingface/label-files/raw/main/imagenet-1k-id2label.json"
        ) as f:
            mapping = json.load(f)
        # This gives us idx -> label, but we need wnid -> idx
        # We'll use the standard ImageNet wnid list
        with urlopen(
            "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        ) as f:
            pass  # Just checking availability
    except Exception:
        pass
    return wnid_to_idx


def get_images_and_labels(val_dir: str) -> list:
    """
    Get list of (image_path, wnid) tuples from ImageNet-style validation folder.
    Folder structure: val_dir/nXXXXXXXX/image.JPEG
    """
    samples = []
    val_path = Path(val_dir)
    for class_dir in sorted(val_path.iterdir()):
        if class_dir.is_dir() and class_dir.name.startswith("n"):
            wnid = class_dir.name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                    samples.append((str(img_file), wnid))
    return samples


def build_wnid_to_idx_from_vit(vit_model) -> dict:
    """
    Build wnid -> idx mapping using ViT model's id2label.
    ViT's id2label contains entries like: {0: "tench, Tinca tinca", ...}
    We need to match these to wnids.
    """
    # Load the standard ImageNet wnid ordering
    from urllib.request import urlopen
    
    # Get the mapping from class index to wnid
    with urlopen(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    ) as f:
        # This file has labels, not wnids
        pass
    
    # Use a direct wnid list (standard ImageNet-1k ordering)
    # The ordering is alphabetical by wnid
    wnid_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    
    # Alternative: use torchvision's mapping
    # For now, we'll build from the folder structure and match by label similarity
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compare ResNet-50 vs ViT-Base on ImageNet validation")
    parser.add_argument("--val-dir", default="imagenet-val-short", help="Path to validation folder")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--output-csv", default="comparison_results.csv", help="Output CSV file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Start total timing
    total_start = perf_counter()

    # Load id2label from local JSON
    id2label_path = os.path.join(args.val_dir, "imagenet-1k-id2label.json")
    if os.path.exists(id2label_path):
        id2label = load_id2label(id2label_path)
    else:
        from urllib.request import urlopen
        with urlopen(
            "https://huggingface.co/datasets/huggingface/label-files/raw/main/imagenet-1k-id2label.json"
        ) as f:
            id2label = json.load(f)

    # Build wnid -> class_idx mapping
    # Standard ImageNet-1k uses alphabetical ordering of wnids
    from urllib.request import urlopen
    wnid_list_url = "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/_meta.py"
    
    # Simpler approach: load wnid ordering directly
    # ImageNet-1k wnids in order (class 0 to 999)
    wnid_order_url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    
    # Actually, let's use a simpler method: load from synset file
    try:
        with urlopen("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json") as f:
            simple_labels = json.load(f)
    except Exception:
        simple_labels = None

    # Build wnid to idx mapping from LOC_synset_mapping.txt style
    # For now, we'll use a hardcoded approach based on folder names
    # The wnid folders in ImageNet val are the synset IDs
    # We need to map them to class indices (0-999)
    
    # Load the official mapping
    wnid_to_idx = {}
    try:
        with urlopen("https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt") as f:
            pass
    except Exception:
        pass

    # Use torchvision's ImageFolder which assigns indices alphabetically
    samples = get_images_and_labels(args.val_dir)
    print(f"Found {len(samples)} images in {args.val_dir}")

    if not samples:
        print("No images found. Check the val-dir path.")
        return

    # Get unique wnids and sort them to match ImageFolder behavior
    unique_wnids = sorted(set(wnid for _, wnid in samples))
    wnid_to_local_idx = {wnid: idx for idx, wnid in enumerate(unique_wnids)}
    
    # For proper ImageNet evaluation, we need the global wnid -> class_idx mapping
    # Load it from a reliable source
    try:
        with urlopen("https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json") as f:
            class_index = json.load(f)
        # class_index is {str(idx): [wnid, label], ...}
        wnid_to_global_idx = {v[0]: int(k) for k, v in class_index.items()}
    except Exception as e:
        print(f"Warning: Could not load wnid mapping: {e}")
        print("Using local folder ordering instead (may not match model's class indices)")
        wnid_to_global_idx = wnid_to_local_idx

    # --- Load ResNet-50 (timm) ---
    # https://huggingface.co/timm/resnet50.a1_in1k
    print("\nLoading timm/resnet50.a1_in1k...")
    resnet_load_start = perf_counter()
    resnet = timm.create_model("resnet50.a1_in1k", pretrained=True)
    resnet.eval()
    resnet.to(device)
    resnet_load_time = (perf_counter() - resnet_load_start) * 1000
    
    resnet_cfg = timm.data.resolve_model_data_config(resnet)
    resnet_transform = timm.data.create_transform(**resnet_cfg, is_training=False)

    # --- Load ViT-Base (HuggingFace) ---
    # https://huggingface.co/google/vit-base-patch16-224
    print("Loading google/vit-base-patch16-224...")
    vit_load_start = perf_counter()
    vit_model_id = "google/vit-base-patch16-224"
    vit_processor = ViTImageProcessor.from_pretrained(vit_model_id, use_fast=True)
    vit = AutoModelForImageClassification.from_pretrained(vit_model_id)
    vit.eval()
    vit.to(device)
    vit_load_time = (perf_counter() - vit_load_start) * 1000
    
    vit_id2label = vit.config.id2label

    # --- Evaluation ---
    print(f"\nEvaluating {len(samples)} images...\n")
    
    results = []
    resnet_correct_top1 = 0
    resnet_correct_top5 = 0
    vit_correct_top1 = 0
    vit_correct_top5 = 0
    
    # Timing lists (all in ms)
    resnet_inference_times = []   # Forward pass only
    vit_inference_times = []      # Forward pass only
    resnet_preprocess_times = []  # Transform/preprocessing
    vit_preprocess_times = []     # Processor call
    image_load_times = []         # PIL image loading
    
    resnet_only_correct = []  # ResNet correct, ViT wrong
    vit_only_correct = []     # ViT correct, ResNet wrong
    both_correct = []
    both_wrong = []
    
    eval_start = perf_counter()

    for img_path, wnid in samples:
        # Get ground truth class index
        gt_idx = wnid_to_global_idx.get(wnid, -1)
        if gt_idx == -1:
            print(f"Warning: Unknown wnid {wnid}, skipping {img_path}")
            continue
        
        gt_label = id2label.get(str(gt_idx), "<unknown>")
        
        # Load image (timed)
        img_load_start = perf_counter()
        img = Image.open(img_path).convert("RGB")
        img_load_time = (perf_counter() - img_load_start) * 1000
        image_load_times.append(img_load_time)
        
        # --- ResNet preprocessing (timed) ---
        resnet_preprocess_start = perf_counter()
        resnet_input = resnet_transform(img).unsqueeze(0).to(device)
        resnet_preprocess_time = (perf_counter() - resnet_preprocess_start) * 1000
        resnet_preprocess_times.append(resnet_preprocess_time)
        
        # --- ResNet inference (timed) ---
        resnet_infer_start = perf_counter()
        with torch.no_grad():
            resnet_logits = resnet(resnet_input)
        resnet_infer_time = (perf_counter() - resnet_infer_start) * 1000
        resnet_inference_times.append(resnet_infer_time)
        
        resnet_probs = resnet_logits.softmax(dim=1)
        resnet_top5_probs, resnet_top5_idx = torch.topk(resnet_probs, k=5, dim=1)
        resnet_pred_idx = resnet_top5_idx[0, 0].item()
        resnet_pred_label = id2label.get(str(resnet_pred_idx), "<unknown>")
        resnet_top1_correct = (resnet_pred_idx == gt_idx)
        resnet_top5_correct = (gt_idx in resnet_top5_idx[0].tolist())
        
        # --- ViT preprocessing (timed) ---
        vit_preprocess_start = perf_counter()
        vit_inputs = vit_processor(images=img, return_tensors="pt")
        vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}
        vit_preprocess_time = (perf_counter() - vit_preprocess_start) * 1000
        vit_preprocess_times.append(vit_preprocess_time)
        
        # --- ViT inference (timed) ---
        vit_infer_start = perf_counter()
        with torch.no_grad():
            vit_logits = vit(**vit_inputs).logits
        vit_infer_time = (perf_counter() - vit_infer_start) * 1000
        vit_inference_times.append(vit_infer_time)
        
        vit_probs = vit_logits.softmax(dim=1)
        vit_top5_probs, vit_top5_idx = torch.topk(vit_probs, k=5, dim=1)
        vit_pred_idx = vit_top5_idx[0, 0].item()
        vit_pred_label = vit_id2label.get(vit_pred_idx, "<unknown>")
        vit_top1_correct = (vit_pred_idx == gt_idx)
        vit_top5_correct = (gt_idx in vit_top5_idx[0].tolist())
        
        # Update counters
        if resnet_top1_correct:
            resnet_correct_top1 += 1
        if resnet_top5_correct:
            resnet_correct_top5 += 1
        if vit_top1_correct:
            vit_correct_top1 += 1
        if vit_top5_correct:
            vit_correct_top5 += 1
        
        # Track disagreements
        img_name = os.path.basename(img_path)
        if resnet_top1_correct and not vit_top1_correct:
            resnet_only_correct.append((img_name, gt_label, resnet_pred_label, vit_pred_label))
        elif vit_top1_correct and not resnet_top1_correct:
            vit_only_correct.append((img_name, gt_label, resnet_pred_label, vit_pred_label))
        elif resnet_top1_correct and vit_top1_correct:
            both_correct.append((img_name, gt_label))
        else:
            both_wrong.append((img_name, gt_label, resnet_pred_label, vit_pred_label))
        
        results.append({
            "image": img_name,
            "wnid": wnid,
            "gt_idx": gt_idx,
            "gt_label": gt_label,
            "resnet_pred_idx": resnet_pred_idx,
            "resnet_pred_label": resnet_pred_label,
            "resnet_top1_correct": resnet_top1_correct,
            "resnet_top5_correct": resnet_top5_correct,
            "resnet_conf": float(resnet_top5_probs[0, 0]) * 100,
            "resnet_preprocess_ms": resnet_preprocess_time,
            "resnet_inference_ms": resnet_infer_time,
            "vit_pred_idx": vit_pred_idx,
            "vit_pred_label": vit_pred_label,
            "vit_top1_correct": vit_top1_correct,
            "vit_top5_correct": vit_top5_correct,
            "vit_conf": float(vit_top5_probs[0, 0]) * 100,
            "vit_preprocess_ms": vit_preprocess_time,
            "vit_inference_ms": vit_infer_time,
            "image_load_ms": img_load_time,
        })

    # --- Summary ---
    eval_time = (perf_counter() - eval_start) * 1000
    total_time = (perf_counter() - total_start) * 1000
    n = len(results)
    
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Compute totals per model
    resnet_total_preprocess = sum(resnet_preprocess_times)
    resnet_total_inference = sum(resnet_inference_times)
    resnet_total_time = resnet_total_preprocess + resnet_total_inference
    
    vit_total_preprocess = sum(vit_preprocess_times)
    vit_total_inference = sum(vit_inference_times)
    vit_total_time = vit_total_preprocess + vit_total_inference
    
    image_load_total = sum(image_load_times)
    
    # Timing summary - SPLIT BY MODEL
    print("\n" + "=" * 80)
    print("TIMING COMPARISON (perf_counter) - ALL TIMES IN MILLISECONDS")
    print("=" * 80)
    
    print(f"\n{'METRIC':<45} {'ResNet (ms)':>15} {'ViT (ms)':>15}")
    print("-" * 80)
    
    # Model loading
    print(f"{'Model load time':<45} {resnet_load_time:>15.2f} {vit_load_time:>15.2f}")
    
    # Total times (all images)
    print(f"\n--- TOTAL TIME ({n} images) [ms] ---")
    print(f"{'Preprocessing (total)':<45} {resnet_total_preprocess:>15.2f} {vit_total_preprocess:>15.2f}")
    print(f"{'Inference (total)':<45} {resnet_total_inference:>15.2f} {vit_total_inference:>15.2f}")
    print(f"{'Preprocess + Inference (total)':<45} {resnet_total_time:>15.2f} {vit_total_time:>15.2f}")
    
    # Per-image averages
    print(f"\n--- PER-IMAGE AVERAGE [ms] ---")
    print(f"{'Preprocessing (avg)':<45} {resnet_total_preprocess/n:>15.2f} {vit_total_preprocess/n:>15.2f}")
    print(f"{'Inference (avg)':<45} {resnet_total_inference/n:>15.2f} {vit_total_inference/n:>15.2f}")
    print(f"{'Preprocess + Inference (avg)':<45} {resnet_total_time/n:>15.2f} {vit_total_time/n:>15.2f}")
    
    # Per-image min/max
    print(f"\n--- PER-IMAGE MIN [ms] ---")
    print(f"{'Preprocessing (min)':<45} {min(resnet_preprocess_times):>15.2f} {min(vit_preprocess_times):>15.2f}")
    print(f"{'Inference (min)':<45} {min(resnet_inference_times):>15.2f} {min(vit_inference_times):>15.2f}")
    
    print(f"\n--- PER-IMAGE MAX [ms] ---")
    print(f"{'Preprocessing (max)':<45} {max(resnet_preprocess_times):>15.2f} {max(vit_preprocess_times):>15.2f}")
    print(f"{'Inference (max)':<45} {max(resnet_inference_times):>15.2f} {max(vit_inference_times):>15.2f}")
    
    # Throughput
    resnet_throughput = 1000.0 / (resnet_total_time/n) if resnet_total_time > 0 else 0
    vit_throughput = 1000.0 / (vit_total_time/n) if vit_total_time > 0 else 0
    print(f"\n--- THROUGHPUT [images/sec] ---")
    print(f"{'Images/second':<45} {resnet_throughput:>15.2f} {vit_throughput:>15.2f}")
    
    # Shared timing (not model-specific)
    print(f"\n--- SHARED TIMING [ms] ---")
    print(f"{'Image loading (PIL) - total':<45} {image_load_total:>15.2f}")
    print(f"{'Image loading (PIL) - avg':<45} {image_load_total/n:>15.2f}")
    print(f"{'Evaluation loop total':<45} {eval_time:>15.2f}")
    print(f"{'Total script time':<45} {total_time:>15.2f}")
    print(f"{'Total script time':<45} {total_time/1000:>15.2f} sec")
    
    # Accuracy summary
    print("\n" + "-" * 80)
    print("ACCURACY")
    print("-" * 80)
    print(f"{'Model':<35} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'Avg Inference':>14}")
    print("-" * 80)
    print(f"{'timm/resnet50.a1_in1k':<35} {resnet_correct_top1/n*100:>11.2f}% {resnet_correct_top5/n*100:>11.2f}% {sum(resnet_inference_times)/n:>12.2f} ms")
    print(f"{'google/vit-base-patch16-224':<35} {vit_correct_top1/n*100:>11.2f}% {vit_correct_top5/n*100:>11.2f}% {sum(vit_inference_times)/n:>12.2f} ms")
    
    print(f"\n{'Category':<40} {'Count':>10}")
    print("-" * 50)
    print(f"{'Both models correct':<40} {len(both_correct):>10}")
    print(f"{'ResNet correct, ViT wrong':<40} {len(resnet_only_correct):>10}")
    print(f"{'ViT correct, ResNet wrong':<40} {len(vit_only_correct):>10}")
    print(f"{'Both models wrong':<40} {len(both_wrong):>10}")
    
    # Show some examples of disagreements
    if resnet_only_correct:
        print("\n--- ResNet correct, ViT wrong (up to 5 examples) ---")
        for img, gt, rn_pred, vit_pred in resnet_only_correct[:5]:
            print(f"  {img}: GT={gt}, ResNet={rn_pred}, ViT={vit_pred}")
    
    if vit_only_correct:
        print("\n--- ViT correct, ResNet wrong (up to 5 examples) ---")
        for img, gt, rn_pred, vit_pred in vit_only_correct[:5]:
            print(f"  {img}: GT={gt}, ResNet={rn_pred}, ViT={vit_pred}")
    
    if both_wrong:
        print("\n--- Both wrong (up to 5 examples) ---")
        for img, gt, rn_pred, vit_pred in both_wrong[:5]:
            print(f"  {img}: GT={gt}, ResNet={rn_pred}, ViT={vit_pred}")

    # Save results to CSV
    import csv
    csv_path = os.path.join(os.path.dirname(args.val_dir) or ".", args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to: {csv_path}")

    # --- Generate comparison plot ---
    plot_path = os.path.join(os.path.dirname(args.val_dir) or ".", "results.png")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ResNet-50 vs ViT-Base Comparison", fontsize=16, fontweight="bold")
    
    models = ["ResNet-50", "ViT-Base"]
    colors = ["#2ecc71", "#3498db"]  # Green for ResNet, Blue for ViT
    
    # 1. Model Load Time
    ax = axes[0, 0]
    load_times = [resnet_load_time, vit_load_time]
    bars = ax.bar(models, load_times, color=colors)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Model Load Time")
    for bar, val in zip(bars, load_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    
    # 2. Inference Time (avg per image)
    ax = axes[0, 1]
    inference_avgs = [resnet_total_inference/n, vit_total_inference/n]
    bars = ax.bar(models, inference_avgs, color=colors)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Avg Inference Time per Image")
    for bar, val in zip(bars, inference_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    
    # 3. Throughput
    ax = axes[0, 2]
    throughputs = [resnet_throughput, vit_throughput]
    bars = ax.bar(models, throughputs, color=colors)
    ax.set_ylabel("Images/sec")
    ax.set_title("Throughput")
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    
    # 4. Accuracy (Top-1 and Top-5)
    ax = axes[1, 0]
    x = np.arange(2)
    width = 0.35
    top1_accs = [resnet_correct_top1/n*100, vit_correct_top1/n*100]
    top5_accs = [resnet_correct_top5/n*100, vit_correct_top5/n*100]
    bars1 = ax.bar(x - width/2, top1_accs, width, label="Top-1", color=colors)
    bars2 = ax.bar(x + width/2, top5_accs, width, label="Top-5", color=[c + "80" for c in colors])  # Lighter
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 105)
    for bar, val in zip(bars1, top1_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, top5_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    
    # 5. Agreement/Disagreement pie chart
    ax = axes[1, 1]
    agreement_data = [len(both_correct), len(resnet_only_correct), len(vit_only_correct), len(both_wrong)]
    agreement_labels = ["Both correct", "ResNet only", "ViT only", "Both wrong"]
    agreement_colors = ["#27ae60", "#2ecc71", "#3498db", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(agreement_data, labels=agreement_labels, autopct="%1.1f%%", colors=agreement_colors, startangle=90)
    ax.set_title("Model Agreement")
    
    # 6. Inference time distribution (box plot)
    ax = axes[1, 2]
    bp = ax.boxplot([resnet_inference_times, vit_inference_times], tick_labels=models, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time Distribution")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
