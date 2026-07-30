# LISA Prompt-Agreement Failure Detector

This repository extends the
[official LISA implementation](https://github.com/dvlab-research/LISA) with
an annotation-free failure detector for reasoning segmentation. Given several
paraphrased prompts for the same image, LISA predicts one mask per prompt and
uses their agreement as its confidence:

$$
s = \min_{i<j} \operatorname{IoU}(M_i, M_j), \qquad
\texttt{unconfident} = (s < \tau).
$$

The comparison below uses the six prompts supplied with each selected
ReasonSeg test image. Pairwise agreement is measured over the full image, so
the detector does not use ground truth. Empty–empty masks have IoU 1 and
empty–nonempty masks have IoU 0. A score equal to the threshold is confident.

## Environment

Inference requires one CUDA GPU. The reproduced run used Python 3.10,
PyTorch 1.13.1 with CUDA 11.7, Transformers 4.31.0, BF16, and
LISA-7B-v1.

Create a `uv` environment and install the original LISA dependencies:

```bash
uv venv --python 3.10 ~/envs/lisa
source ~/envs/lisa/bin/activate
uv pip install -r requirements.txt
```

## Model and data

Download
[LISA-7B-v1](https://huggingface.co/xinlai/LISA-7B-v1) into
`weights/LISA-7B-v1`.

Download the ReasonSeg test set and place its 779 image/annotation pairs in
`test`:

```text
LISA/
├── test/
│   ├── <sample>.jpg
│   ├── <sample>.json
│   └── ...
└── weights/
    └── LISA-7B-v1/
```

The evaluator selects the 366 annotations containing six prompts, then
excludes three annotations with duplicate prompts and one with an empty
target. The resulting evaluation cohort contains 362 images and 2,172 model
calls.

Validate the dataset without loading the model:

```bash
python evaluate_failure_detector.py \
  --dataset-dir test \
  --validate-data-only
```

## Run

Run a one-image smoke test first:

```bash
python evaluate_failure_detector.py \
  --dataset-dir test \
  --version weights/LISA-7B-v1 \
  --output-dir runs/failure_detector_smoke \
  --precision bf16 \
  --limit 1 \
  --resume
```

Run the complete comparison:

```bash
python evaluate_failure_detector.py \
  --dataset-dir test \
  --version weights/LISA-7B-v1 \
  --output-dir runs/failure_detector \
  --precision bf16 \
  --resume
```

The script uses local model files by default. Add `--allow-network` if the
CLIP vision tower or other required Hugging Face files are not already
cached.

Inference is checkpointed after every image. Repeating the same command with
`--resume` skips completed records. To regenerate the tables without loading
LISA, run:

```bash
python evaluate_failure_detector.py \
  --dataset-dir test \
  --version weights/LISA-7B-v1 \
  --output-dir runs/failure_detector \
  --precision bf16 \
  --summarize-only
```

Thresholds 0.15 and 0.30 are evaluated by default. Supply repeated
`--threshold` options to evaluate other thresholds without rerunning
inference:

```bash
python evaluate_failure_detector.py \
  --dataset-dir test \
  --version weights/LISA-7B-v1 \
  --output-dir runs/failure_detector \
  --precision bf16 \
  --summarize-only \
  --threshold 0.10 \
  --threshold 0.20
```

## Outputs

The output directory contains:

- `run_config.json`: model, data, prompt, and inference configuration;
- `records/<sample>.json`: resumable sample-level predictions and metrics;
- `per_sample.jsonl`: all 15 pairwise mask IoUs and prompt-0 metrics;
- `threshold_comparison.csv`: machine-readable comparison;
- `threshold_comparison.json`: comparison plus metric metadata;
- `threshold_comparison.md`: report-ready table.

Prompt 0 is the baseline prediction. For each retained or excluded cohort:

- **gIoU** is the mean prompt-0 IoU across images;
- **cIoU** is total prompt-0 intersection divided by total union;
- **coverage** is the fraction with minimum pairwise IoU at least the
  threshold.

ReasonSeg pixels labeled 255 are ignored when computing accuracy. They are
not ignored when measuring prediction agreement because the detector must
remain annotation-free.

## Reproduced comparison

| Threshold | Retained | Coverage | Retained gIoU | Retained cIoU | Excluded | Excluded gIoU | Excluded cIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 362 | 1.000000 | 0.468056 | 0.494602 | 0 | — | — |
| 0.15 | 275 | 0.759669 | 0.518082 | 0.523244 | 87 | 0.309928 | 0.408262 |
| 0.30 | 241 | 0.665746 | 0.558492 | 0.561506 | 121 | 0.287931 | 0.370646 |

Higher thresholds reject more samples. The retained metrics therefore
describe selective accuracy and must be interpreted together with coverage.

## Model API

The existing single-prompt `evaluate()` method is unchanged.
`LISAForCausalLM.evaluate_prompt_set()` runs several prompts for one image and
returns the detector result:

```python
result = model.evaluate_prompt_set(
    images_clip,
    images,
    input_ids_list,
    resize_list,
    original_size_list,
    confidence_threshold=0.30,
    tokenizer=tokenizer,
)

score = result["minimum_pairwise_iou"]
unconfident = result["unconfident"]
pairwise_iou = result["pairwise_mask_iou"]
```

The method also returns the generated token IDs and predicted masks for each
prompt.
