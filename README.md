# PulmoFoundation

**PulmoFoundation** is a foundation model specifically designed for lung pathology whole-slide image analysis. Built on the Virchow2 architecture with LoRA continual pretraining, it provides state-of-the-art feature embeddings for computational pathology applications in lung cancer and respiratory disease research.

## Features

- **Lung-Specific**: Fine-tuned on extensive lung pathology datasets

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch = 2.0.1
- peft = 0.17.1
- time = 1.0.11
- Pillow = 11.0.0 
- opencv-python = 4.10.0.84

### Install Dependencies

```bash
git clone https://github.com/dddavid4real/PulmoFoundation
cd PulmoFoundation
pip install -r requirements.txt
```

### Download Model Checkpoint

🌟 The pretrained checkpoint of PulmoFoundation is provided here: https://huggingface.co/david4real/PulmoFoundation.

Download the pre-trained checkpoint and place it in `models/ckpts/`:

```bash
mkdir -p models/ckpts
# Download PulmoFoundation-E2.pth from HuggingFace to models/ckpts/
```

## Quick Start

### Basic Usage

```python
from models import get_model, get_transform
from PIL import Image

# Load model and preprocessing pipeline
model = get_model('cuda', 'models/ckpts/PulmoFoundation-E2.pth')
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
from models import get_model, get_transform
from PIL import Image
import torch

model = get_model('cuda', 'models/ckpts/PulmoFoundation-E2.pth')
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
- Inspired by [CLAM](https://github.com/mahmoodlab/CLAM) for WSI processing
- Uses [PEFT](https://github.com/huggingface/peft) for efficient continual pretraining

## Version History

- **v1.0.0** (2025-12): Initial release with PulmoFoundation checkpoint
