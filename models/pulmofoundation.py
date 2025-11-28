"""
PulmoFoundation: A Foundation Model for Lung Pathology
Based on Virchow2 architecture with LoRA fine-tuning
"""
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


def get_transform():
    """
    Get the image transformation pipeline for PulmoFoundation.
    
    Returns:
        torchvision.transforms: Transformation pipeline for preprocessing images
    """
    # Create a temporary model to get the transform configuration
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2", 
        pretrained=True, 
        mlp_layer=SwiGLUPacked, 
        act_layer=torch.nn.SiLU
    )
    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    del model
    return transforms


def get_model(device, checkpoint_path):
    """
    Load PulmoFoundation model from checkpoint.
    
    Args:
        device (str or torch.device): Device to load model on (e.g., 'cuda' or 'cpu')
        checkpoint_path (str): Path to the model checkpoint (.pth file)
    
    Returns:
        callable: A function that takes an image tensor and returns embeddings
        
    Example:
        >>> model = get_model('cuda', 'models/ckpts/PulmoFoundation-E2.pth')
        >>> img = transform(Image.open('image.jpg'))
        >>> img = img.unsqueeze(0).cuda()  # Add batch dimension
        >>> features = model(img)  # Returns [N, 2560] embeddings
    """
    from peft import LoraConfig, get_peft_model
    
    try:
        # Create base Virchow2 model
        # print("Loading base Virchow2 model...")
        base_model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked, 
            act_layer=torch.nn.SiLU
        )
        
        # Configure LoRA parameters matching training setup
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "attn.qkv",
                "attn.proj",
            ],
            lora_dropout=0.1,
        )
        
        # Apply LoRA to base model
        model = get_peft_model(base_model, config)
        
        # Load checkpoint
        print(f"Loading PulmoFoundation checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['teacher']
        
        # Remove 'backbone.' prefix from state dict keys
        corrected_state_dict = {}
        prefix_to_remove = "backbone."
        
        for key, value in state_dict.items():
            if key.startswith(prefix_to_remove):
                new_key = key[len(prefix_to_remove):]
                corrected_state_dict[new_key] = value
            else:
                corrected_state_dict[key] = value
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(corrected_state_dict, strict=False)
        
        if missing_keys:
            # print(f"Info: {len(missing_keys)} missing keys (expected for LoRA)")
            pass
        if unexpected_keys:
            pass
            # print(f"Info: {len(unexpected_keys)} unexpected keys (likely classification head)")
        
        # Merge LoRA weights into base model
        # print("Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        merged_model = merged_model.to(device)
        merged_model.eval()
        
        print("PulmoFoundation model loaded successfully!")
        
        # Create inference function
        def inference_func(image):
            """
            Extract features from input images.
            
            Args:
                image (torch.Tensor): Preprocessed image tensor [N, 3, H, W]
            
            Returns:
                torch.Tensor: Feature embeddings [N, 2560]
                    - First 1280 dims: class token
                    - Last 1280 dims: averaged patch tokens
            """
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output = merged_model(image)
            
            class_token = output[:, 0]        # [N, 1280]
            patch_tokens = output[:, 5:]       # [N, 256, 1280] (skip register tokens 1-4)
            
            # Concatenate class token and mean-pooled patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [N, 2560]
            return embedding
        
        return inference_func
        
    except Exception as e:
        print(f"Error loading PulmoFoundation model: {e}")
        raise

