# TSSA

Crop disease image-text retrieval with spatial relation-aware and temporal evidence-aware modeling.

## Setup

Hardware configuration: NVIDIA GeForce RTX 3090. The implementation uses PyTorch with BF16 mixed precision.

Python 3.11. Install a CUDA-enabled PyTorch build suitable for your system, then install the dependencies:

```bash
pip install -r requirements.txt
```

Set the Swin-T checkpoint and BERT-base-uncased location in `configs/main.yaml`. The Swin checkpoint uses the official `swin_tiny_patch4_window7_224_22k.pth` format. BERT can be a local pretrained directory or the `bert-base-uncased` model identifier.

## Main configuration

| Setting | Value |
| --- | --- |
| Visual / textual encoders | Pretrained Swin-T / BERT-base-uncased |
| Image processing | 224 × 224; random resized crop and horizontal flip for training |
| Maximum text length | 80 tokens |
| Embedding dimension / attention heads / dropout | 512 / 8 / 0.1 |
| Training epochs / image batch size | 50 / 128, with three descriptions per image |
| Optimizer | AdamW; β = (0.9, 0.98); weight decay = 0.01 |
| Encoder learning rate | 2 × 10⁻⁵ |
| Projection and refinement learning rate | 1 × 10⁻⁴ |
| Learning-rate schedule | Five-epoch linear warm-up, then cosine decay |
| Initial learnable temperature | 0.07 |
| Semantic neighborhood size k / refinement steps T | 6 / 4 |
| Class-center loss weight λca / margin m | 0.5 / 0.2 |
| Initial context coefficient η = softplus(θ) | 1.0, jointly optimized |
| Class-center loss weight warm-up | First five epochs |
| Default random seed | 42; additional main runs use 43 and 44 |

Within each mini-batch, every represented category contains at least two distinct images. Each training image is used once per epoch without class oversampling. Model selection uses validation mR.

## Input

Set image and annotation paths in `configs/main.yaml`. Each split is a JSON list with one record per image:

```json
{"image": "relative/path.jpg", "image_id": 1, "label": 0, "caption": ["description 1", "description 2", "description 3"]}
```

Keep all descriptions of an image in one split. Training labels are contiguous integers starting from zero. Each training category needs at least two images. Images, annotations and pretrained or trained weights are supplied separately.

## Train

```bash
python train.py --config configs/main.yaml --seed 42 --test-after-training
```

The checkpoint with the highest validation mR is selected for testing. Seeds 43 and 44 use the same configuration with `--seed 43` or `--seed 44`.

## Inference and evaluation

```bash
python infer.py --config configs/main.yaml --checkpoint outputs/main/seed42/checkpoint_best.pt --annotations data/test.json
python evaluate.py --scores outputs/inference/scores.npz --annotations data/test.json
```

Evaluation uses the full candidate pool. An image has its three paired descriptions as positive targets; each description has its source image as its positive target. mR is the mean of the six bidirectional Recall values. Per-class and class-macro results use the labels in the annotation file.
