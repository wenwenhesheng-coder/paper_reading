import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch

from roman_r import ROMAN_R


def read_frames(dataset_root, start_frame, frame_count, max_side):
    input_dir = os.path.join(dataset_root, "input")
    gt_dir = os.path.join(dataset_root, "groundtruth")
    if frame_count == 0:
        with open(os.path.join(dataset_root, "temporalROI.txt"), "r", encoding="utf-8") as stream:
            roi_start, roi_end = [int(value) for value in stream.readline().split()]
        start_frame = roi_start if start_frame is None else start_frame
        frame_count = roi_end - start_frame + 1
    frames = []
    targets = []

    for frame_number in range(start_frame, start_frame + frame_count):
        input_path = os.path.join(input_dir, "in{:06d}.jpg".format(frame_number))
        target_path = os.path.join(gt_dir, "gt{:06d}.png".format(frame_number))
        frame = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if frame is None or target is None:
            raise FileNotFoundError("Missing frame or ground truth: {}".format(frame_number))

        if max(frame.shape) > max_side:
            scale = max_side / max(frame.shape)
            size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, size, interpolation=cv2.INTER_NEAREST)

        frames.append(frame.astype(np.float32) / 255.0)
        targets.append(target)

    return np.stack(frames), np.stack(targets)


def compute_f1(prediction, target):
    valid = np.logical_and(target != 85, target != 170)
    target_motion = target == 255
    predicted_motion = prediction >= 0.5
    tp = np.count_nonzero(valid & target_motion & predicted_motion)
    fp = np.count_nonzero(valid & ~target_motion & predicted_motion)
    fn = np.count_nonzero(valid & target_motion & ~predicted_motion)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * precision * recall / (precision + recall + 1e-7)


def compute_frame_metrics(prediction, target, frame_number):
    valid = np.logical_and(target != 85, target != 170)
    target_motion = target == 255
    predicted_motion = prediction >= 0.5
    tp = int(np.count_nonzero(valid & target_motion & predicted_motion))
    tn = int(np.count_nonzero(valid & ~target_motion & ~predicted_motion))
    fp = int(np.count_nonzero(valid & ~target_motion & predicted_motion))
    fn = int(np.count_nonzero(valid & target_motion & ~predicted_motion))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return {
        "frame": frame_number,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_results(output_dir, frames, backgrounds, masks, targets):
    os.makedirs(output_dir, exist_ok=True)
    for index in range(frames.shape[0]):
        input_image = np.clip(frames[index] * 255.0, 0, 255).astype(np.uint8)
        background_image = np.clip(backgrounds[index] * 255.0, 0, 255).astype(np.uint8)
        mask_image = np.clip(masks[index] * 255.0, 0, 255).astype(np.uint8)
        prediction_image = ((masks[index] >= 0.5) * 255).astype(np.uint8)
        target_image = ((targets[index] == 255) * 255).astype(np.uint8)
        comparison = cv2.hconcat([input_image, background_image, mask_image, prediction_image, target_image])
        cv2.imwrite(os.path.join(output_dir, "frame_{:03d}.png".format(index + 1)), comparison)


def save_metrics(output_dir, metrics):
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="..\\dataset\\baseline\\highway")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="..\\results\\highway_all")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--frames", type=int, default=0, help="0 means all frames in temporalROI.txt")
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden-filters", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *unused_args, **unused_kwargs: self

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(script_dir, dataset_root)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    dataset_root = os.path.abspath(dataset_root)
    output_dir = os.path.abspath(output_dir)
    if args.start_frame is None:
        with open(os.path.join(dataset_root, "temporalROI.txt"), "r", encoding="utf-8") as stream:
            args.start_frame = int(stream.readline().split()[0])
    frames, targets = read_frames(dataset_root, args.start_frame, args.frames, args.max_side)
    params = {
        "layers": args.layers,
        "kernel": [(1, 5)] * args.layers,
        "hidden_filters": args.hidden_filters,
        "coef_L": 0.8,
        "coef_S": 0.05,
        "coef_S_side": 0.025,
        "l1_l1": False,
        "reweightedl1_l1": True,
        "l1_l2": False,
        "img_size": None,
    }
    net = ROMAN_R(params)
    checkpoint_loaded = False
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(state)
        checkpoint_loaded = True

    net = net.to(device)
    net.eval()
    data_chunks = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for chunk_start in range(0, frames.shape[0], args.chunk_size):
            chunk = torch.from_numpy(frames[chunk_start:chunk_start + args.chunk_size]).unsqueeze(0).to(device)
            data_chunks.append(net(chunk))
    elapsed = time.perf_counter() - start_time

    backgrounds = np.concatenate([item[0][0].detach().cpu().numpy() for item in data_chunks], axis=0)
    masks = np.concatenate([item[2][0].detach().cpu().numpy() for item in data_chunks], axis=0)
    save_results(output_dir, frames, backgrounds, masks, targets)
    metrics = [
        compute_frame_metrics(masks[index], targets[index], args.start_frame + index)
        for index in range(frames.shape[0])
    ]
    save_metrics(output_dir, metrics)
    summary = {
        "dataset": dataset_root,
        "start_frame": args.start_frame,
        "frame_count": frames.shape[0],
        "input_shape": [1, frames.shape[0], frames.shape[1], frames.shape[2]],
        "resized_shape": list(frames.shape[1:]),
        "checkpoint_loaded": checkpoint_loaded,
        "elapsed_seconds": elapsed,
        "mask_min": float(masks.min()),
        "mask_max": float(masks.max()),
        "mask_mean": float(masks.mean()),
        "f1_at_0.5": float(compute_f1(masks, targets)),
        "mean_per_frame_f1": float(np.mean([item["f1"] for item in metrics])),
        "worst_frame_f1": float(np.min([item["f1"] for item in metrics])),
        "best_frame_f1": float(np.max([item["f1"] for item in metrics])),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
