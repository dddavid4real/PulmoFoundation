import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Path to CHIEF weights
CHIEF_WEIGHT_DIR = os.path.dirname(os.path.abspath(__file__))


def initialize_weights(module):
    """Initialize weights for linear and batch norm layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net_Gated(nn.Module):
    """Gated Attention Network for MIL aggregation."""
    
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class CHIEFEncoder(nn.Module):
    """
    CHIEF slide encoder - attention-based aggregation with organ embeddings.
    
    Based on: Wang et al., "CHIEF: A Clinical Histopathology Imaging Evaluation 
    Foundation Model for Cancer Diagnosis and Prognosis Prediction", Nature 2024.
    """
    
    def __init__(self, size_arg="small", dropout=True):
        super(CHIEFEncoder, self).__init__()
        
        # Size configurations: [input_dim, hidden_dim, attn_dim]
        self.size_dict = {
            'xs': [384, 256, 256], 
            "small": [768, 512, 256], 
            "big": [1024, 512, 384], 
            'large': [2048, 1024, 512]
        }
        size = self.size_dict[size_arg]
        self.hidden_dim = size[1]
        
        # Feature projection + attention network
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # Organ embedding for anatomical conditioning
        self.register_buffer('organ_embedding', torch.randn(19, 768))
        
        # Text-to-vision projection
        self.text_to_vision = nn.Sequential(
            nn.Linear(768, size[1]), 
            nn.ReLU(), 
            nn.Dropout(p=0.25)
        )
        
        initialize_weights(self)
        
        # Load organ embeddings
        text_embed_path = os.path.join(CHIEF_WEIGHT_DIR, 'Text_emdding.pth')
        if os.path.exists(text_embed_path):
            word_embedding = torch.load(text_embed_path, map_location='cpu')
            self.organ_embedding.data = word_embedding.float()
            print(f"[CHIEF] Loaded organ embeddings from {text_embed_path}")
        else:
            print(f"[CHIEF] Warning: Text_emdding.pth not found at {text_embed_path}")
    
    def forward(self, h, anatomical_index):
        """
        Forward pass through CHIEF encoder.
        
        Args:
            h: Patch features of shape [N, D] or [1, N, D]
            anatomical_index: Organ index (0-18), Lung = 1
            
        Returns:
            WSI_feature: Slide-level embedding [1, hidden_dim]
            attention: Attention weights [1, N]
        """
        # Handle batch dimension
        if len(h.shape) == 3:
            h = h.squeeze(0)  # [1, N, D] -> [N, D]
        
        h_ori = h
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        
        # Get organ embedding
        if isinstance(anatomical_index, int):
            anatomical_index = torch.tensor([anatomical_index], device=h.device)
        embed_batch = self.organ_embedding[anatomical_index]
        embed_batch = self.text_to_vision(embed_batch)
        
        # Aggregate features
        WSI_feature = torch.mm(A, h)  # [1, hidden_dim]
        slide_embeddings = torch.mm(A, h_ori)  # [1, input_dim]
        
        # Add anatomical context
        M = WSI_feature + embed_batch
        
        return M, A_raw


class ChiefClassifier(nn.Module):
    """
    CHIEF classifier with frozen slide encoder and trainable classification head.
    
    Uses Lung anatomical index (1) as built-in setting.
    """
    
    def __init__(
        self, 
        n_classes, 
        n_features=768,
        freeze_encoder=True,
        size_arg="small",
        dropout=0.25,
        act="relu"
    ):
        super(ChiefClassifier, self).__init__()
        
        self.n_features = n_features
        self.freeze_encoder = freeze_encoder
        
        # Anatomical index for Lung (built-in)
        self.anatomical_index = 6  # Lung
        
        # Load CHIEF encoder
        print("[CHIEF] Loading CHIEF slide encoder...")
        self.slide_encoder = CHIEFEncoder(size_arg=size_arg, dropout=True)
        
        # Get hidden dimension
        hidden_dim = self.slide_encoder.hidden_dim  # 512 for "small"
        
        # Load pretrained weights
        weight_path = os.path.join(CHIEF_WEIGHT_DIR, 'CHIEF_pretraining.pth')
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu')
            # Filter for encoder keys only
            encoder_keys = {k: v for k, v in state_dict.items() 
                          if k.startswith('attention_net') or 
                             k.startswith('text_to_vision') or
                             k.startswith('organ_embedding')}
            missing, unexpected = self.slide_encoder.load_state_dict(encoder_keys, strict=False)
            print(f"[CHIEF] Loaded pretrained weights from {weight_path}")
            if missing:
                print(f"[CHIEF] Missing keys: {missing[:5]}..." if len(missing) > 5 else f"[CHIEF] Missing keys: {missing}")
        else:
            print(f"[CHIEF] Warning: Pretrained weights not found at {weight_path}")
        
        # Freeze encoder parameters if requested
        if freeze_encoder:
            print("[CHIEF] Freezing slide encoder parameters...")
            for param in self.slide_encoder.parameters():
                param.requires_grad = False
            print("[CHIEF] Slide encoder frozen. Only classification head is trainable.")
        
        # Trainable classification head (following official design)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        print(f"[CHIEF] Classifier head: {hidden_dim} -> {n_classes}")
        print(f"[CHIEF] Using anatomical index: {self.anatomical_index} (Lung)")
    
    def forward(self, features, return_attn=False):
        """
        Forward pass through the CHIEF model for survival prediction.
        
        Args:
            features: Tile embeddings of shape [1, N, D] or [N, D]
            return_attn: Whether to return attention weights
        
        Returns:
            hazards: Hazard probabilities per time bin [1, n_classes]
            S: Cumulative survival probability [1, n_classes]
        """
        # Forward through slide encoder
        if self.freeze_encoder:
            with torch.no_grad():
                slide_embedding, attention = self.slide_encoder(features, self.anatomical_index)
        else:
            slide_embedding, attention = self.slide_encoder(features, self.anatomical_index)
        
        # Classification -> Survival output
        logits = self.classifier(slide_embedding)
        
        # Convert to hazards and survival probability (survival format)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S

