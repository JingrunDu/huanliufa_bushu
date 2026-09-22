"""Generate semantic weight maps for converter-valve (换流阀) guided reconstruction.

Instead of a binary person-exclusion mask, this produces a continuous per-pixel
weight map that drives semantic-weighted 3DGS training. Pixels fall into three
mutually exclusive classes:

    - person / dynamic distractor  -> weight 0.0   (excluded from loss)
    - building background          -> weight 1.0   (floor / wall / ceiling)
    - equipment                    -> weight 2.0   (everything that is neither
                                                    person nor background)

Equipment is obtained by subtraction (full image minus person minus background)
rather than by prompting for equipment directly. This avoids both the
over-generalization of generic prompts like "metal structure" (which can flood
the whole frame) and the missed detections of part-level jargon prompts.

The weight is stored in a single-channel PNG using the linear encoding
    pixel_value = round(weight * WEIGHT_SCALE)        (WEIGHT_SCALE = 100)
so training reads it back as  weight = pixel_value / 100.0 .

Two products are written per image:
    1. semantic_weights/L2PRO/{camera}/{stem}.png   -> the weight map used in training
    2. semantic_weights_vis/L2PRO/{camera}/{stem}.jpg -> a color overlay for eyeballing

Usage:
    # small batch validation (first 50 training images)
    python generate_semantic_weights.py --camera camera_0 --training-only --limit 50

    # full training set
    python generate_semantic_weights.py --camera camera_0 --training-only
    python generate_semantic_weights.py --camera camera_1 --training-only

Run from the sam3 directory so that `import sam3` works, e.g.:
    cd /home/dujingrun.djr/code/3dgs/sam3
    /tmp/fmc/sam3/bin/python /home/dujingrun.djr/code/3dgs/FastGS/generate_semantic_weights.py \
        --camera camera_0 --training-only --limit 50
"""

import argparse
import os
import struct
import time

import cv2
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WEIGHT_SCALE = 100.0  # pixel_value = weight * WEIGHT_SCALE, clamped to [0, 255]

# Three-class scheme. Equipment is NOT prompted; it is whatever remains after
# removing person and building background. Background prompts segment the
# warehouse/hall envelope (floor, wall, ceiling).
PERSON_PROMPT = "person"
BACKGROUND_PROMPTS = ["floor", "wall", "ceiling"]

PERSON_WEIGHT = 0.0       # excluded from loss
BACKGROUND_WEIGHT = 1.0   # baseline
EQUIPMENT_WEIGHT = 2.0    # high-weight region for guided reconstruction

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate semantic weight maps with SAM3")
    parser.add_argument("--camera", type=str, default="camera_0",
                        choices=["camera_0", "camera_1"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N images (0 = all)")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="SAM3 confidence threshold")
    parser.add_argument("--base-dir", type=str,
                        default="/home/dujingrun.djr/code/3dgs/FastGS/data/perspective")
    parser.add_argument("--training-only", action="store_true",
                        help="Only process images listed in sparse/0/images.bin")
    parser.add_argument("--name-list", type=str, default=None,
                        help="Path to a txt file of COLMAP image names "
                             "(e.g. test_split.txt); only those of the current "
                             "--camera are processed")
    parser.add_argument("--dilate", type=int, default=3,
                        help="Dilate each semantic mask by N pixels (0 to disable)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip color visualization overlays")
    parser.add_argument("--checkpoint", type=str,
                        default="/data/model/sam3_weights/facebook/sam3.1/sam3.1_multiplex.pt")
    return parser.parse_args()

def read_training_images(base_dir):
    """Read image names from COLMAP images.bin."""
    images_bin = os.path.join(base_dir, "sparse/0/images.bin")
    if not os.path.exists(images_bin):
        return None
    images = set()
    with open(images_bin, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            struct.unpack("<I", f.read(4))
            f.read(32 + 24 + 4)
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n2d * 24)
            images.add(name)
    return images

def dilate_mask(mask, pixels):
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)

def collect_prompt_mask(processor, state, prompt, img_h, img_w, dilate_px):
    """Run one text prompt and return a merged binary mask at image resolution."""
    output = processor.set_text_prompt(state=state, prompt=prompt)
    masks = output["masks"]
    merged = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask_tensor in masks:
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.shape != (img_h, img_w):
            mask_np = cv2.resize(
                mask_np.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, (mask_np > 0).astype(np.uint8))
    merged = dilate_mask(merged, dilate_px)
    return merged

def build_weight_map(processor, state, img_h, img_w, dilate_px):
    """Compose the three-class weight map via subtraction.

    equipment = full image - person - building background
    """
    person_mask = collect_prompt_mask(
        processor, state, PERSON_PROMPT, img_h, img_w, dilate_px)

    background_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for prompt in BACKGROUND_PROMPTS:
        layer_mask = collect_prompt_mask(
            processor, state, prompt, img_h, img_w, dilate_px)
        background_mask = np.maximum(background_mask, layer_mask)

    # Start with everything as equipment, then carve out background and person.
    # Person has the highest priority and overrides background where they overlap.
    weight = np.full((img_h, img_w), EQUIPMENT_WEIGHT, dtype=np.float32)
    weight[background_mask > 0] = BACKGROUND_WEIGHT
    weight[person_mask > 0] = PERSON_WEIGHT

    return weight

def encode_weight_png(weight):
    """Encode float weight map to uint8 PNG values via linear scaling."""
    encoded = np.clip(weight * WEIGHT_SCALE, 0, 255).astype(np.uint8)
    return encoded

def make_visualization(image_bgr, weight):
    """Color overlay: red=person(0), blue=background(1.0), green=equipment(2.0)."""
    overlay = image_bgr.copy()
    alpha = 0.45

    layers = [
        (np.isclose(weight, PERSON_WEIGHT), (0, 0, 255)),       # red   = person
        (np.isclose(weight, BACKGROUND_WEIGHT), (255, 80, 0)),  # blue  = background
        (np.isclose(weight, EQUIPMENT_WEIGHT), (0, 200, 0)),    # green = equipment
    ]
    for region, color in layers:
        for channel in range(3):
            overlay[..., channel][region] = (
                alpha * color[channel]
                + (1 - alpha) * overlay[..., channel][region]
            ).astype(np.uint8)

    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(overlay, "red=person  blue=background  green=equipment",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return overlay

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    images_dir = os.path.join(args.base_dir, "images", "L2PRO", args.camera)
    output_dir = os.path.join(
        args.base_dir, "semantic_weights", "L2PRO", args.camera)
    vis_dir = os.path.join(
        args.base_dir, "semantic_weights_vis", "L2PRO", args.camera)
    os.makedirs(output_dir, exist_ok=True)
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)

    all_images = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))

    if args.training_only:
        training_set = read_training_images(args.base_dir)
        if training_set:
            prefix = f"L2PRO/{args.camera}/"
            training_names = {
                name.replace(prefix, "") for name in training_set
                if name.startswith(prefix)
            }
            all_images = sorted(f for f in all_images if f in training_names)
            print(f"Filtered to {len(all_images)} training images")

    if args.name_list:
        with open(args.name_list) as handle:
            wanted = set(line.strip() for line in handle if line.strip())
        prefix = f"L2PRO/{args.camera}/"
        wanted_names = {
            name.replace(prefix, "") for name in wanted
            if name.startswith(prefix)
        }
        all_images = sorted(f for f in all_images if f in wanted_names)
        print(f"Filtered to {len(all_images)} images from {args.name_list} "
              f"for {args.camera}")

    if args.limit > 0:
        all_images = all_images[:args.limit]

    print(f"Camera: {args.camera}, processing {len(all_images)} images")

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    print("SAM3 loaded!")

    start_time = time.time()

    for i, img_name in enumerate(all_images):
        img_path = os.path.join(images_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        weight_path = os.path.join(output_dir, stem + ".png")

        pil_image = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_image.size

        state = processor.set_image(pil_image)
        weight = build_weight_map(processor, state, img_h, img_w, args.dilate)

        encoded = encode_weight_png(weight)
        cv2.imwrite(weight_path, encoded)

        if not args.no_vis:
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            vis = make_visualization(image_bgr, weight)
            cv2.imwrite(os.path.join(vis_dir, stem + ".jpg"), vis)

        if (i + 1) % 10 == 0 or (i + 1) == len(all_images):
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (len(all_images) - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1}/{len(all_images)}] {speed:.1f} img/s, "
                  f"ETA: {eta/60:.1f} min")

    elapsed = time.time() - start_time
    print(f"\nDone! {len(all_images)} images in {elapsed/60:.1f} min")
    print(f"  Weight maps: {output_dir}")
    if not args.no_vis:
        print(f"  Visualizations: {vis_dir}")

if __name__ == "__main__":
    main()
