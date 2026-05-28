# PulmoFoundation

**PulmoFoundation** is a foundation model specifically designed for lung pathology whole-slide image analysis. Built on the Virchow2 architecture with LoRA continual pretraining, it provides feature embeddings for computational pathology applications in lung cancer and respiratory disease research.

## Installation

### Prerequisites
- Request access to the Virchow2 model on [Hugging Face](https://huggingface.co/paige-ai/Virchow2).

### Install Dependencies

```bash
git clone https://github.com/dddavid4real/PulmoFoundation
cd PulmoFoundation
pip install -r requirements.txt
pip install -e .
```

The editable install registers the importable `model_loading` package, so examples can use `from model_loading import get_model, get_transform` from outside the repository directory as well. The downstream diagnosis and survival folders are script-based workflows and should be run from their own directories.

### Download Model Checkpoint and Features

The pretrained checkpoint and public TCGA-NSCLC feature archive are provided through the [PulmoFoundation Hugging Face repository](https://huggingface.co/david4real/PulmoFoundation):

```text
PulmoFoundation-E2.pth        # encoder checkpoint
TCGA__NSCLC.z01/.z02/.zip     # public TCGA feature archive
```

Download the pretrained checkpoint and place it in `model_loading/ckpts/`:

```bash
mkdir -p model_loading/ckpts
# Download PulmoFoundation-E2.pth from Hugging Face to model_loading/ckpts/
```

## Repository Layout

```text
PulmoFoundation/
  model_loading/              # PulmoFoundation encoder and transforms
  diagnosis_and_prediction/   # MIL diagnosis, molecular prediction, and external evaluation
  survival_analysis/          # Survival MIL, C-index, and risk scores
```

For WSI preprocessing, including coordinate extraction, patch cropping, and feature extraction, use [PrePATH](https://github.com/birkhoffkiki/PrePATH/tree/main). This repository starts from extracted patch features for downstream diagnosis and survival workflows.

See [diagnosis_and_prediction](diagnosis_and_prediction/) and [survival_analysis](survival_analysis/) for the released downstream MIL workflows.

## Workflow Overview

```text
Raw WSI
  -> PrePATH tissue detection, tiling, and patch extraction
  -> PulmoFoundation-E2 patch feature extraction
  -> Slide-level feature tensors in TCGA__NSCLC/pt_files/
  -> ABMIL diagnosis and molecular prediction
  -> AttMIL survival analysis
```

## Public TCGA-NSCLC Feature Package

For reviewer convenience, we provide pre-extracted TCGA-NSCLC PulmoFoundation-E2 features through the [PulmoFoundation Hugging Face repository](https://huggingface.co/david4real/PulmoFoundation).

Download all three archive parts into the same directory:

```text
TCGA__NSCLC.z01
TCGA__NSCLC.z02
TCGA__NSCLC.zip
```

Then unzip from the `.zip` file:

```bash
unzip TCGA__NSCLC.zip
```

The extracted folder should have this structure:

```text
TCGA__NSCLC/
  patches/
  pt_files/
    PulmoFoundation-E2/
      *.pt
```

The downstream examples use `pt_files/`. The `patches/` folder is included for transparency and inspection.

Downstream diagnosis and survival scripts expect the feature root to be:

```bash
FEATURE_ROOT=/path/to/TCGA__NSCLC/pt_files
```

The same `TCGA__NSCLC/pt_files` feature folder is used for the released TCGA NSCLC subtyping, EGFR/STK11 molecular prediction, and LUAD/LUSC survival examples.

## Released Downstream Examples

| Task type | Task | CSV file | Feature root |
|---|---|---|---|
| Diagnosis | TCGA NSCLC subtyping | `diagnosis_and_prediction/dataset_csv/External_TCGA_NSCLC.csv` | `TCGA__NSCLC/pt_files` |
| Molecular prediction | TCGA EGFR | `diagnosis_and_prediction/dataset_csv/External_TCGA_EGFR.csv` | `TCGA__NSCLC/pt_files` |
| Molecular prediction | TCGA STK11 | `diagnosis_and_prediction/dataset_csv/TCGA_STK11.csv` | `TCGA__NSCLC/pt_files` |
| Survival | TCGA-LUAD OS | `survival_analysis/dataset_csv/LUAD.csv` | `TCGA__NSCLC/pt_files` |
| Survival | TCGA-LUSC OS | `survival_analysis/dataset_csv/LUSC.csv` | `TCGA__NSCLC/pt_files` |

## Quick Start

### Basic Usage

```python
from model_loading import get_model, get_transform
from PIL import Image

# Load model and preprocessing pipeline
model = get_model('cuda', 'model_loading/ckpts/PulmoFoundation-E2.pth')
transform = get_transform()

# Load and preprocess image
img = Image.open('path/to/your/image.jpg')  # Prefer 512x512 patches at 40X
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [1, 3, H, W]
img_tensor = img_tensor.cuda()  # Move to GPU

# Extract features
features = model(img_tensor)  # Shape: [1, 2560]
print(f"Feature shape: {features.shape}")
print(f"Feature vector: {features}")
```

### Batch Processing

```python
from model_loading import get_model, get_transform
from PIL import Image
import torch

model = get_model('cuda', 'model_loading/ckpts/PulmoFoundation-E2.pth')
transform = get_transform()

# Process multiple images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
images = [transform(Image.open(p)) for p in image_paths]
batch = torch.stack(images).cuda()  # Shape: [N, 3, H, W]

# Extract features for all images at once
features = model(batch)  # Shape: [N, 2560]
```

## Reproducing Diagnosis and Molecular Prediction Examples

Run commands from `diagnosis_and_prediction/`.

Set the feature root:

```bash
cd diagnosis_and_prediction
FEATURE_ROOT=/path/to/TCGA__NSCLC/pt_files
```

Evaluate TCGA NSCLC subtyping:

```bash
python main.py \
  --model ABMIL \
  --study External_TCGA_NSCLC \
  --root ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --csv_file dataset_csv/External_TCGA_NSCLC.csv \
  --evaluate \
  --resume './results/results_42/NSCLC/[ABMIL]' \
  --tqdm
```

Evaluate TCGA EGFR prediction:

```bash
python main.py \
  --model ABMIL \
  --study External_TCGA_EGFR \
  --root ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --csv_file dataset_csv/External_TCGA_EGFR.csv \
  --evaluate \
  --resume './results/results_42/EGFR/[ABMIL]' \
  --tqdm
```

Evaluation results are saved under the selected checkpoint directory.

## Reproducing Survival Analysis Examples

Run commands from `survival_analysis/`.

Set the feature root:

```bash
cd survival_analysis
FEATURE_ROOT=/path/to/TCGA__NSCLC/pt_files
```

Train TCGA-LUAD survival model:

```bash
python main.py \
  --model AttMIL \
  --csv_file ./dataset_csv/LUAD.csv \
  --feature_path ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --study LUAD \
  --modal WSI \
  --num_epoch 20 \
  --batch_size 1 \
  --lr 2e-4
```

Train TCGA-LUSC survival model:

```bash
python main.py \
  --model AttMIL \
  --csv_file ./dataset_csv/LUSC.csv \
  --feature_path ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --study LUSC \
  --modal WSI \
  --num_epoch 20 \
  --batch_size 1 \
  --lr 2e-4
```

Survival outputs are written to:

```text
survival_analysis/results/WSI/<TASK>/...
```

Survival examples are released as training workflows from TCGA features. Trained survival checkpoints are not bundled in this release.

## CSV and Feature Naming Conventions

Diagnosis CSV files use this schema:

```text
case,slide,label,fold
```

For diagnosis, the `slide` column should not include `.pt`; the loader appends `.pt` internally.

Survival CSV files use this schema:

```text
Study,ID,Event,Status,WSI,split
```

For survival, the `WSI` column should include the `.pt` suffix.

If one case has multiple slides, join slide names with `;`.

## Expected Outputs

Diagnosis training writes outputs to:

```text
diagnosis_and_prediction/results/
```

Diagnosis external evaluation writes result CSVs and prediction files under the selected checkpoint directory.

Survival training writes C-index summaries, bootstrap intervals, and risk outputs to:

```text
survival_analysis/results/WSI/<TASK>/...
```

## Scope of This Release

This repository releases:

- PulmoFoundation model loading code.
- Downstream diagnosis and molecular prediction workflows.
- Downstream survival analysis workflows.
- Public TCGA-NSCLC feature tensors for reviewer testing.
- Public TCGA CSV manifests.
- Released ABMIL checkpoints for TCGA NSCLC and EGFR evaluation.
- Survival training workflows for TCGA-LUAD and TCGA-LUSC.

Private institutional slides, private feature tensors, and private annotation files are not included.

## Acknowledgments

- Built on [Virchow2](https://huggingface.co/paige-ai/Virchow2) by Paige AI
- WSI preprocessing is handled by [PrePATH](https://github.com/birkhoffkiki/PrePATH/tree/main)
- Uses [PEFT](https://github.com/huggingface/peft) for efficient continual pretraining

## Version History

- **v1.1.0** (2026-05): Update with PulmoFoundation Downstream Evaluation Workflow
- **v1.0.0** (2025-12): Initial release with PulmoFoundation checkpoint
