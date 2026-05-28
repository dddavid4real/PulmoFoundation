import importlib

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def _load_slide_encoder():
    try:
        return importlib.import_module("gigapath.slide_encoder")
    except ImportError as exc:
        raise ImportError(
            "GigaPath support requires the optional GigaPath slide encoder package. "
            "Install it or add the GigaPath repository to PYTHONPATH before using "
            "--model Gigapath."
        ) from exc


class GigapathClassifier(nn.Module):
    """
    GigaPath classifier with frozen slide encoder and trainable classification head.
    
    Based on official GigaPath usage:
        slide_encoder.eval()
        with torch.no_grad():
            output = slide_encoder(tile_embed, coordinates).squeeze()
    """
    
    def __init__(
        self, 
        n_classes, 
        n_features=1536, 
        freeze_encoder=True,
        model_arch="gigapath_slide_enc12l768d",
        dropout=0.25,
        act="relu"
    ):
        super(GigapathClassifier, self).__init__()
        
        self.n_features = n_features
        self.freeze_encoder = freeze_encoder
        
        # Load pretrained slide encoder from HuggingFace
        print("[GigaPath] Loading pretrained slide encoder from HuggingFace...")
        slide_encoder = _load_slide_encoder()
        self.slide_encoder = slide_encoder.create_model(
            pretrained="hf_hub:prov-gigapath/prov-gigapath",
            model_arch=model_arch,
            in_chans=n_features
        )
        
        # Freeze encoder parameters if requested
        if freeze_encoder:
            print("[GigaPath] Freezing slide encoder parameters...")
            for param in self.slide_encoder.parameters():
                param.requires_grad = False
            print("[GigaPath] Slide encoder frozen. Only classification head is trainable.")
        
        # Get embedding dimension from the slide encoder
        if "768d" in model_arch:
            embed_dim = 768
        elif "1024d" in model_arch:
            embed_dim = 1024
        elif "1536d" in model_arch:
            embed_dim = 1536
        else:
            embed_dim = 768
        
        # Trainable classification head (following official design: embed_dim -> n_classes)
        self.classifier = nn.Linear(embed_dim, n_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Max patches for uniform sampling
        #* 18000 max patches for full fine-tuning
        #* 160000 max patches for frozen encoder
        # self.max_patches = 18000
        self.max_patches = 160000
        
        print(f"[GigaPath] Classifier head: {embed_dim} -> {n_classes}")
    
    def forward(self, features, coords, return_attn=False):
        """
        Forward pass through the GigaPath model.
        
        Args:
            features: Tile embeddings of shape [1, N, D] or [N, D]
            coords: Tile coordinates of shape [1, N, 2] or [N, 2] - raw pixel coordinates
            return_attn: Whether to return attention weights (not supported)
        
        Returns:
            logits: Classification logits of shape [1, n_classes]
        """
        # Ensure batch dimension
        if len(features.shape) == 2:
            features = features.unsqueeze(0)  # [N, D] -> [1, N, D]
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)  # [N, 2] -> [1, N, 2]
        
        # Uniform sampling if number of patches exceeds max_patches
        n_patches = features.shape[1]
        if n_patches > self.max_patches:
            # Generate uniformly spaced indices
            indices = torch.linspace(0, n_patches - 1, steps=self.max_patches).long()
            indices = indices.to(features.device)
            features = features[:, indices, :]
            coords = coords[:, indices, :]
        
        # Use autocast for mixed precision (like official GigaPath usage)
        with autocast():
            # Forward through slide encoder (following official usage pattern)
            self.slide_encoder.eval()
            if self.freeze_encoder:
                with torch.no_grad():
                    embeddings = self.slide_encoder(features, coords)
            else:
                embeddings = self.slide_encoder(features, coords)
            
            # embeddings is a list, take the last output
            slide_embedding = embeddings[-1]
            
            # Classification
            logits = self.classifier(slide_embedding)
        
        if return_attn:
            return logits, None
        else:
            return logits
