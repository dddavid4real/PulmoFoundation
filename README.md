# PulmoFoundation

**PulmoFoundation** is a foundation model specifically designed for lung pathology whole-slide image analysis. Built on the Virchow2 architecture with LoRA continual pretraining, it provides state-of-the-art feature embeddings for computational pathology applications in lung cancer and respiratory disease research.

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

### Download Model Checkpoint

The pretrained checkpoint of PulmoFoundation is provided [here](https://huggingface.co/david4real/PulmoFoundation).

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

## Acknowledgments

- Built on [Virchow2](https://huggingface.co/paige-ai/Virchow2) by Paige AI
- WSI preprocessing is handled by [PrePATH](https://github.com/birkhoffkiki/PrePATH/tree/main)
- Uses [PEFT](https://github.com/huggingface/peft) for efficient continual pretraining

## Version History

- **v1.0.0** (2025-12): Initial release with PulmoFoundation checkpoint
