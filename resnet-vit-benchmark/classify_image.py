import argparse
import json
from time import perf_counter
from urllib.request import urlopen

import torch
from PIL import Image
import timm
from transformers import ViTImageProcessor, AutoModelForImageClassification


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", "--image", dest="image_path", default="test.png")
    args = parser.parse_args()

    image_path = args.image_path

    # Load image
    img = load_image(image_path)

    # Load ImageNet-1k id2label mapping from Hugging Face (for timm ResNet-50)
    with urlopen(
        "https://huggingface.co/datasets/huggingface/label-files/raw/main/imagenet-1k-id2label.json"
    ) as f:
        id2label = json.load(f)

    # --- Inference with timm/resnet50.a1_in1k ---
    # Note: this usage follows the model card example at
    # https://huggingface.co/timm/resnet50.a1_in1k
    resnet_model = timm.create_model("resnet50.a1_in1k", pretrained=True)
    resnet_model.eval()

    # Get model-specific transforms (resize, normalization, etc.)
    resnet_data_config = timm.data.resolve_model_data_config(resnet_model)
    resnet_transform = timm.data.create_transform(**resnet_data_config, is_training=False)

    # Preprocess image and create batch of size 1
    resnet_input = resnet_transform(img).unsqueeze(0)

    # Inference timing for timm/resnet50.a1_in1k (forward pass + softmax/topk)
    start = perf_counter()
    with torch.no_grad():
        resnet_output = resnet_model(resnet_input)
        resnet_probabilities = resnet_output.softmax(dim=1) * 100.0
        resnet_top5_prob, resnet_top5_indices = torch.topk(resnet_probabilities, k=5, dim=1)
    end = perf_counter()
    resnet_elapsed_ms = (end - start) * 1000.0

    print(f"Top-5 predictions timm/resnet50.a1_in1k (label, probability %) - inference: {resnet_elapsed_ms:.2f} ms")
    for prob, idx in zip(resnet_top5_prob[0], resnet_top5_indices[0]):
        class_index = int(idx)
        label = id2label.get(str(class_index), "<unknown>")
        print(f"{label:50s} {float(prob):6.2f} %")

    # --- Inference with google/vit-base-patch16-224 ---
    # Note: ViTImageProcessor is used explicitly here to match the model card docs;
    # https://huggingface.co/google/vit-base-patch16-224
    # AutoImageProcessor.from_pretrained(vit_model_id, use_fast=True) would also work
    # by resolving to the same processor class.
    vit_model_id = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(vit_model_id, use_fast=True)
    vit_model = AutoModelForImageClassification.from_pretrained(vit_model_id)
    vit_model.eval()

    vit_inputs = processor(images=img, return_tensors="pt")

    # Inference timing for google/vit-base-patch16-224 (forward pass only)
    start = perf_counter()
    with torch.no_grad():
        vit_output = vit_model(**vit_inputs).logits
        vit_probabilities = vit_output.softmax(dim=1) * 100.0
        vit_top5_prob, vit_top5_indices = torch.topk(vit_probabilities, k=5, dim=1)
    end = perf_counter()
    vit_elapsed_ms = (end - start) * 1000.0

    id2label_vit = vit_model.config.id2label

    print(f"\nTop-5 predictions google/vit-base-patch16-224 (label, probability %) - inference: {vit_elapsed_ms:.2f} ms")
    for prob, idx in zip(vit_top5_prob[0], vit_top5_indices[0]):
        class_index = int(idx)
        label = id2label_vit.get(class_index, "<unknown>")
        print(f"{label:50s} {float(prob):6.2f} %")


if __name__ == "__main__":
    main()

# python3 classify_image.py --img test4.png
