#!/usr/bin/env python3
"""Evaluate LISA's prompt-agreement failure detector on ReasonSeg."""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROMPT_COUNT = 6
PAIR_COUNT = PROMPT_COUNT * (PROMPT_COUNT - 1) // 2
EXPECTED_SOURCE_COUNT = 779
EXPECTED_SIX_PROMPT_COUNT = 366
EXPECTED_SAMPLE_COUNT = 362
DEFAULT_THRESHOLDS = (0.15, 0.30)
SEGMENTATION_SUFFIX = "Please output segmentation mask."


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run six-prompt LISA inference and evaluate prompt-agreement "
            "failure detection on ReasonSeg"
        )
    )
    parser.add_argument("--version", default="xinlai/LISA-7B-v1")
    parser.add_argument("--dataset-dir", default="test")
    parser.add_argument(
        "--output-dir", default="runs/failure_detector"
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=("fp32", "bf16", "fp16"),
    )
    parser.add_argument("--image-size", default=1024, type=int)
    parser.add_argument("--model-max-length", default=512, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument(
        "--vision-tower",
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--conv-type",
        default="llava_v1",
        choices=("llava_v1", "llava_llama_2"),
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help=(
            "failure threshold; repeat to compare values "
            "(default: 0.15 and 0.30)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate the first N samples; intended for smoke tests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-sample records",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="regenerate tables from completed records without loading LISA",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="validate and describe the ReasonSeg cohort without inference",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow Transformers to download missing model files",
    )
    return parser.parse_args(argv)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
    )


def normalize_space(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def read_annotation(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    doubled = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    return abs(doubled) / 2.0


def has_valid_target(annotation):
    for shape in annotation.get("shapes", []):
        label = normalize_space(shape.get("label", "")).casefold()
        if label == "flag" or "ignore" in label:
            continue
        if polygon_area(shape.get("points", [])) > 0:
            return True
    return False


def append_segmentation_suffix(text):
    text = normalize_space(text)
    if text and text[-1] not in ".?!":
        text += "."
    if text.casefold().endswith(SEGMENTATION_SUFFIX.casefold()):
        return text
    return "{} {}".format(text, SEGMENTATION_SUFFIX)


def discover_samples(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    annotation_paths = sorted(dataset_dir.glob("*.json"))
    image_paths = sorted(dataset_dir.glob("*.jpg"))
    if len(annotation_paths) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            "expected {} JSON annotations in {}, found {}".format(
                EXPECTED_SOURCE_COUNT,
                dataset_dir,
                len(annotation_paths),
            )
        )
    if len(image_paths) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            "expected {} JPG images in {}, found {}".format(
                EXPECTED_SOURCE_COUNT,
                dataset_dir,
                len(image_paths),
            )
        )

    text_counts = Counter()
    samples = []
    exclusions = []
    for annotation_path in annotation_paths:
        annotation = read_annotation(annotation_path)
        texts = [
            normalize_space(text)
            for text in annotation.get("text", [])
        ]
        text_counts[len(texts)] += 1
        if len(texts) != PROMPT_COUNT:
            continue

        sample_id = annotation_path.stem
        image_path = annotation_path.with_suffix(".jpg")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if any(not text for text in texts):
            raise ValueError(
                "{} contains an empty prompt".format(sample_id)
            )
        if not annotation.get("is_sentence"):
            raise ValueError(
                "{} has six prompts but is_sentence is false".format(
                    sample_id
                )
            )

        normalized = [text.casefold() for text in texts]
        if len(set(normalized)) != PROMPT_COUNT:
            exclusions.append(
                {"sample_id": sample_id, "reason": "duplicate_prompt"}
            )
            continue
        if not has_valid_target(annotation):
            exclusions.append(
                {"sample_id": sample_id, "reason": "empty_ground_truth"}
            )
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "prompts": [
                    append_segmentation_suffix(text) for text in texts
                ],
            }
        )

    if text_counts[PROMPT_COUNT] != EXPECTED_SIX_PROMPT_COUNT:
        raise ValueError(
            "expected {} six-prompt annotations, found {}".format(
                EXPECTED_SIX_PROMPT_COUNT,
                text_counts[PROMPT_COUNT],
            )
        )
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "expected {} valid samples, found {}".format(
                EXPECTED_SAMPLE_COUNT, len(samples)
            )
        )
    if Counter(item["reason"] for item in exclusions) != {
        "duplicate_prompt": 3,
        "empty_ground_truth": 1,
    }:
        raise ValueError(
            "unexpected six-prompt exclusions: {}".format(exclusions)
        )

    return {
        "dataset_dir": str(dataset_dir),
        "source_count": len(annotation_paths),
        "text_count_distribution": dict(sorted(text_counts.items())),
        "six_prompt_count": text_counts[PROMPT_COUNT],
        "sample_count": len(samples),
        "exclusions": sorted(
            exclusions, key=lambda item: item["sample_id"]
        ),
        "samples": sorted(samples, key=lambda item: item["sample_id"]),
    }


def normalize_thresholds(values):
    values = DEFAULT_THRESHOLDS if values is None else tuple(values)
    thresholds = []
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be finite values in [0, 1]")
        if value in thresholds:
            raise ValueError("thresholds must be unique")
        thresholds.append(float(value))
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return tuple(sorted(thresholds))


def select_samples(cohort, limit):
    samples = cohort["samples"]
    if limit is None:
        return samples
    if not 1 <= limit <= len(samples):
        raise ValueError(
            "--limit must be between 1 and {}".format(len(samples))
        )
    return samples[:limit]


def configure_determinism(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def load_model(args):
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor

    from model.LISA import LISAForCausalLM
    from model.segment_anything.utils.transforms import ResizeLongestSide

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LISA inference")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expected one visible GPU, found {}".format(
                torch.cuda.device_count()
            )
        )

    local_only = not args.allow_network
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=local_only,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_ids = tokenizer(
        "[SEG]", add_special_tokens=False
    ).input_ids
    if len(seg_token_ids) != 1:
        raise ValueError("[SEG] must map to exactly one token")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.precision]
    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_ids[0],
        torch_dtype=dtype,
        local_files_only=local_only,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model = model.to(dtype=dtype, device="cuda")
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=dtype, device="cuda")
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.eval()

    clip_processor = CLIPImageProcessor.from_pretrained(
        model.config.vision_tower,
        local_files_only=local_only,
    )
    transform = ResizeLongestSide(args.image_size)
    return model, tokenizer, clip_processor, transform, dtype


def preprocess_sample(sample, clip_processor, transform, dtype, image_size):
    import cv2
    import torch
    import torch.nn.functional as torch_functional

    from utils.data_processing import get_mask_from_json

    image_bgr = cv2.imread(sample["image_path"])
    if image_bgr is None:
        raise ValueError(
            "could not read {}".format(sample["image_path"])
        )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ground_truth, _, _ = get_mask_from_json(
        sample["annotation_path"], image_rgb
    )

    image_clip = clip_processor.preprocess(
        image_rgb, return_tensors="pt"
    )["pixel_values"].to(dtype=dtype, device="cuda")
    resized = transform.apply_image(image_rgb)
    resize = resized.shape[:2]
    image = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .contiguous()
        .float()
    )
    pixel_mean = torch.tensor(
        [123.675, 116.28, 103.53], dtype=torch.float32
    ).view(-1, 1, 1)
    pixel_std = torch.tensor(
        [58.395, 57.12, 57.375], dtype=torch.float32
    ).view(-1, 1, 1)
    image = (image - pixel_mean) / pixel_std
    height, width = image.shape[-2:]
    image = torch_functional.pad(
        image, (0, image_size - width, 0, image_size - height)
    )
    image = image.unsqueeze(0).to(dtype=dtype, device="cuda")
    return {
        "ground_truth": ground_truth.astype(np.uint8),
        "image_clip": image_clip,
        "image": image,
        "resize_list": [resize],
        "original_size_list": [image_rgb.shape[:2]],
    }


def build_input_ids(prompt_text, tokenizer, conv_type):
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.utils import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
    )

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    prompt = prompt.replace(
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN
        + DEFAULT_IMAGE_TOKEN
        + DEFAULT_IM_END_TOKEN,
    )
    conversation = conversation_lib.conv_templates[conv_type].copy()
    conversation.messages = []
    conversation.append_message(conversation.roles[0], prompt)
    conversation.append_message(conversation.roles[1], "")
    rendered = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        rendered, tokenizer, return_tensors="pt"
    ).unsqueeze(0)
    return input_ids.to(device="cuda")


def prediction_config(args, cohort, samples):
    return {
        "schema_version": 1,
        "dataset_dir": cohort["dataset_dir"],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "prompt_count": PROMPT_COUNT,
        "segmentation_suffix": SEGMENTATION_SUFFIX,
        "version": str(Path(args.version).resolve())
        if Path(args.version).exists()
        else args.version,
        "precision": args.precision,
        "image_size": args.image_size,
        "model_max_length": args.model_max_length,
        "max_new_tokens": args.max_new_tokens,
        "vision_tower": args.vision_tower,
        "conv_type": args.conv_type,
        "seed": args.seed,
        "greedy_decoding": True,
    }


def initialize_output(output_dir, config, resume, summarize_only):
    output_dir = Path(output_dir)
    config_path = output_dir / "run_config.json"
    if summarize_only:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("prediction_config") != config:
            raise ValueError(
                "saved run configuration does not match the command"
            )
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("prediction_config") != config:
            raise ValueError(
                "output directory belongs to a different prediction run"
            )
        if not resume:
            raise FileExistsError(
                "{} exists; pass --resume to reuse it".format(
                    config_path
                )
            )
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "{} is nonempty and has no run_config.json".format(
                    output_dir
                )
            )
        atomic_write_json(
            config_path,
            {
                "created_at_utc": utc_now(),
                "prediction_config": config,
            },
        )
    (output_dir / "records").mkdir(parents=True, exist_ok=True)
    return output_dir


def record_path(output_dir, sample_id):
    return output_dir / "records" / (sample_id + ".json")


def binary_accuracy(prediction, ground_truth):
    prediction = np.asarray(prediction, dtype=bool)
    valid = ground_truth != 255
    target = ground_truth == 1
    if not target[valid].any():
        raise ValueError("ground-truth target is empty")
    intersection = int(
        np.logical_and(prediction, target)[valid].sum()
    )
    union = int(np.logical_or(prediction, target)[valid].sum())
    if union <= 0:
        raise ValueError("ground-truth union is empty")
    return {
        "iou": intersection / union,
        "intersection": intersection,
        "union": union,
    }


def infer_sample(
    model,
    tokenizer,
    clip_processor,
    transform,
    dtype,
    sample,
    args,
    detector_threshold,
):
    prepared = preprocess_sample(
        sample,
        clip_processor,
        transform,
        dtype,
        args.image_size,
    )
    input_ids_list = [
        build_input_ids(prompt, tokenizer, args.conv_type)
        for prompt in sample["prompts"]
    ]
    result = model.evaluate_prompt_set(
        prepared["image_clip"],
        prepared["image"],
        input_ids_list,
        prepared["resize_list"],
        prepared["original_size_list"],
        confidence_threshold=detector_threshold,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
    )

    matrix = np.asarray(
        result["pairwise_mask_iou"], dtype=np.float64
    )
    if matrix.shape != (PROMPT_COUNT, PROMPT_COUNT):
        raise ValueError(
            "{} returned agreement matrix {}".format(
                sample["sample_id"], matrix.shape
            )
        )
    minimum = float(matrix[np.triu_indices(PROMPT_COUNT, k=1)].min())
    if not np.isclose(
        minimum,
        result["minimum_pairwise_iou"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("model returned an inconsistent confidence score")
    if result["unconfident"] != (minimum < detector_threshold):
        raise ValueError("model returned an inconsistent failure flag")

    primary_masks = result["primary_masks"]
    prompt0 = np.asarray(primary_masks[0], dtype=bool)
    accuracy = binary_accuracy(
        prompt0, prepared["ground_truth"]
    )
    mask_pixels = [
        int(np.asarray(mask, dtype=bool).sum())
        for mask in primary_masks
    ]
    mask_counts = [
        int(prompt_masks[0].shape[0])
        for prompt_masks in result["pred_masks"]
    ]
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sample_id": sample["sample_id"],
        "prompts": sample["prompts"],
        "pairwise_mask_iou": matrix.tolist(),
        "minimum_pairwise_iou": minimum,
        "model_detector_threshold": detector_threshold,
        "model_unconfident": bool(result["unconfident"]),
        "prompt0_iou": float(accuracy["iou"]),
        "prompt0_intersection": accuracy["intersection"],
        "prompt0_union": accuracy["union"],
        "predicted_pixels_by_prompt": mask_pixels,
        "segmentation_masks_by_prompt": mask_counts,
    }


def run_inference(
    args,
    samples,
    output_dir,
    detector_threshold,
):
    import torch
    from tqdm import tqdm

    configure_determinism(args.seed)
    model, tokenizer, clip_processor, transform, dtype = load_model(args)
    completed = 0
    for sample in tqdm(samples, desc="ReasonSeg"):
        path = record_path(output_dir, sample["sample_id"])
        if path.exists():
            if not args.resume:
                raise FileExistsError(path)
            completed += 1
            continue
        record = infer_sample(
            model,
            tokenizer,
            clip_processor,
            transform,
            dtype,
            sample,
            args,
            detector_threshold,
        )
        atomic_write_json(path, record)
        completed += 1
        torch.cuda.empty_cache()
    return completed


def validate_record(record, sample):
    if record.get("schema_version") != 1:
        raise ValueError(
            "{} has an unexpected schema".format(sample["sample_id"])
        )
    if record.get("sample_id") != sample["sample_id"]:
        raise ValueError(
            "{} has the wrong sample ID".format(sample["sample_id"])
        )
    if record.get("prompts") != sample["prompts"]:
        raise ValueError(
            "{} prompts do not match the dataset".format(
                sample["sample_id"]
            )
        )
    matrix = np.asarray(record.get("pairwise_mask_iou"), dtype=float)
    if matrix.shape != (PROMPT_COUNT, PROMPT_COUNT):
        raise ValueError(
            "{} has an invalid pairwise matrix".format(
                sample["sample_id"]
            )
        )
    if not np.isfinite(matrix).all() or not (
        (0.0 <= matrix).all() and (matrix <= 1.0).all()
    ):
        raise ValueError(
            "{} has out-of-range pairwise IoUs".format(
                sample["sample_id"]
            )
        )
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(
            "{} pairwise matrix is not symmetric".format(
                sample["sample_id"]
            )
        )
    if not np.allclose(
        np.diag(matrix), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "{} pairwise matrix has invalid diagonal".format(
                sample["sample_id"]
            )
        )
    values = matrix[np.triu_indices(PROMPT_COUNT, k=1)]
    if len(values) != PAIR_COUNT:
        raise ValueError("pair-count invariant failed")
    minimum = float(record["minimum_pairwise_iou"])
    if not np.isclose(minimum, values.min(), rtol=0.0, atol=1e-12):
        raise ValueError(
            "{} has an inconsistent confidence score".format(
                sample["sample_id"]
            )
        )
    intersection = int(record["prompt0_intersection"])
    union = int(record["prompt0_union"])
    iou = float(record["prompt0_iou"])
    if intersection < 0 or union <= 0 or intersection > union:
        raise ValueError(
            "{} has invalid accuracy counts".format(sample["sample_id"])
        )
    if not np.isclose(iou, intersection / union, rtol=0.0, atol=1e-12):
        raise ValueError(
            "{} has inconsistent prompt-0 IoU".format(
                sample["sample_id"]
            )
        )
    return record


def load_records(output_dir, samples):
    records = []
    missing = []
    for sample in samples:
        path = record_path(output_dir, sample["sample_id"])
        if not path.is_file():
            missing.append(sample["sample_id"])
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append(validate_record(record, sample))
    if missing:
        raise ValueError(
            "{} samples are incomplete; first missing sample: {}".format(
                len(missing), missing[0]
            )
        )
    return records


def cohort_metrics(records):
    if not records:
        return {"count": 0, "giou": None, "ciou": None}
    intersections = sum(
        int(record["prompt0_intersection"]) for record in records
    )
    unions = sum(int(record["prompt0_union"]) for record in records)
    return {
        "count": len(records),
        "giou": float(
            np.mean([record["prompt0_iou"] for record in records])
        ),
        "ciou": intersections / unions,
    }


def analyze(records, thresholds):
    rows = []
    all_metrics = cohort_metrics(records)
    rows.append(
        {
            "threshold": "none",
            "retained": all_metrics["count"],
            "coverage": 1.0,
            "retained_giou": all_metrics["giou"],
            "retained_ciou": all_metrics["ciou"],
            "excluded": 0,
            "excluded_giou": None,
            "excluded_ciou": None,
        }
    )
    for threshold in thresholds:
        retained = [
            record
            for record in records
            if record["minimum_pairwise_iou"] >= threshold
        ]
        excluded = [
            record
            for record in records
            if record["minimum_pairwise_iou"] < threshold
        ]
        retained_metrics = cohort_metrics(retained)
        excluded_metrics = cohort_metrics(excluded)
        rows.append(
            {
                "threshold": "{:.2f}".format(threshold),
                "retained": retained_metrics["count"],
                "coverage": len(retained) / len(records),
                "retained_giou": retained_metrics["giou"],
                "retained_ciou": retained_metrics["ciou"],
                "excluded": excluded_metrics["count"],
                "excluded_giou": excluded_metrics["giou"],
                "excluded_ciou": excluded_metrics["ciou"],
            }
        )
    return rows


def format_metric(value):
    return "—" if value is None else "{:.6f}".format(value)


def write_outputs(output_dir, records, rows, thresholds):
    atomic_write_text(
        output_dir / "per_sample.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
    )

    fieldnames = [
        "threshold",
        "retained",
        "coverage",
        "retained_giou",
        "retained_ciou",
        "excluded",
        "excluded_giou",
        "excluded_ciou",
    ]
    csv_path = output_dir / "threshold_comparison.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    atomic_write_json(
        output_dir / "threshold_comparison.json",
        {
            "created_at_utc": utc_now(),
            "sample_count": len(records),
            "prompt_count": PROMPT_COUNT,
            "pair_count_per_sample": PAIR_COUNT,
            "confidence_domain": "full_image",
            "confidence_score": "minimum_pairwise_mask_iou",
            "threshold_rule": "unconfident if score < threshold",
            "thresholds": list(thresholds),
            "accuracy_mask": "ReasonSeg ground truth excluding label 255",
            "rows": rows,
        },
    )

    lines = [
        "# Prompt-agreement failure detector",
        "",
        "- Samples: {}".format(len(records)),
        "- Prompts per image: {}".format(PROMPT_COUNT),
        "- Pairwise IoUs per image: {}".format(PAIR_COUNT),
        "",
        "| Threshold | Retained | Coverage | Retained gIoU | "
        "Retained cIoU | Excluded | Excluded gIoU | Excluded cIoU |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {threshold} | {retained} | {coverage} | "
            "{retained_giou} | {retained_ciou} | {excluded} | "
            "{excluded_giou} | {excluded_ciou} |".format(
                threshold=row["threshold"],
                retained=row["retained"],
                coverage=format_metric(row["coverage"]),
                retained_giou=format_metric(row["retained_giou"]),
                retained_ciou=format_metric(row["retained_ciou"]),
                excluded=row["excluded"],
                excluded_giou=format_metric(row["excluded_giou"]),
                excluded_ciou=format_metric(row["excluded_ciou"]),
            )
        )
    atomic_write_text(
        output_dir / "threshold_comparison.md",
        "\n".join(lines) + "\n",
    )


def run(args):
    thresholds = normalize_thresholds(args.thresholds)
    cohort = discover_samples(args.dataset_dir)
    samples = select_samples(cohort, args.limit)
    print(
        "ReasonSeg: {} source images, {} six-prompt annotations, "
        "{} valid samples".format(
            cohort["source_count"],
            cohort["six_prompt_count"],
            cohort["sample_count"],
        )
    )
    print(
        "Selected {} samples ({} model calls).".format(
            len(samples), len(samples) * PROMPT_COUNT
        )
    )
    if args.validate_data_only:
        print(
            "Text-count distribution: {}".format(
                cohort["text_count_distribution"]
            )
        )
        print("Exclusions: {}".format(cohort["exclusions"]))
        return 0

    config = prediction_config(args, cohort, samples)
    output_dir = initialize_output(
        args.output_dir,
        config,
        resume=args.resume,
        summarize_only=args.summarize_only,
    )
    if not args.summarize_only:
        completed = run_inference(
            args,
            samples,
            output_dir,
            detector_threshold=max(thresholds),
        )
        print("Completed {}/{} samples.".format(completed, len(samples)))

    records = load_records(output_dir, samples)
    rows = analyze(records, thresholds)
    write_outputs(output_dir, records, rows, thresholds)
    print(
        "Wrote comparison tables to {}".format(output_dir.resolve())
    )
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
