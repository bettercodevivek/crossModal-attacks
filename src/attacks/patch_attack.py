"""
Universal Adversarial Patch Attack for CLIP.

This attack generates a universal adversarial patch that can be applied
to any image to fool CLIP into misclassifying it as a target caption.
"""
import random
import torch
from torch import nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import get_device


class PatchAttack:
    """
    Universal adversarial patch attack.
    
    Generates a patch that maximizes CLIP similarity with a target caption.
    """
    
    def __init__(self, model, processor, target_text, device=None):
        """
        Initialize patch attack.
        
        Args:
            model: CLIP model
            processor: CLIP processor
            target_text: Target caption to maximize similarity with
            device: Device to run on (auto-detected if None)
        """
        self.model = model
        self.processor = processor
        self.target_text = target_text
        self.device = device if device else get_device()
        
        # Precompute target text embedding
        with torch.no_grad():
            text_inputs = processor(
                text=[target_text], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            self.target_emb = model.get_text_features(**text_inputs)
            self.target_emb = self.target_emb / self.target_emb.norm(dim=-1, keepdim=True)
    
    def generate_patch(self, train_images, patch_size=None, steps=None, 
                      batch_size=None, lr=None, patch_location=None):
        """
        Generate universal adversarial patch.
        
        Args:
            train_images: List of training image tensors
            patch_size: Size of patch (default from Config)
            steps: Number of optimization steps (default from Config)
            batch_size: Batch size for training (default from Config)
            lr: Learning rate (default from Config)
            patch_location: "random" or "topleft" (default from Config)
            
        Returns:
            Learned patch tensor (3, patch_size, patch_size)
        """
        patch_size = patch_size or Config.PATCH_SIZE
        steps = steps or Config.PATCH_STEPS
        batch_size = batch_size or Config.PATCH_BATCH_SIZE
        lr = lr or Config.PATCH_LR
        patch_location = patch_location or Config.PATCH_LOCATION
        
        # Initialize learnable patch
        patch = nn.Parameter(
            torch.rand(3, patch_size, patch_size, device=self.device) * 0.5
        )
        optimizer = torch.optim.Adam([patch], lr=lr)
        
        # Training loop
        for step in range(steps):
            self.model.zero_grad()
            
            # Sample random batch
            batch_indices = random.sample(
                range(len(train_images)), 
                min(batch_size, len(train_images))
            )
            batch = torch.stack([train_images[i] for i in batch_indices]).to(self.device)
            
            # Apply patch
            patched = batch.clone()
            for i in range(patched.size(0)):
                if patch_location == "random":
                    x = random.randint(0, patched.shape[-1] - patch_size)
                    y = random.randint(0, patched.shape[-2] - patch_size)
                else:
                    x = y = 0
                patched[i, :, y:y+patch_size, x:x+patch_size] = patch
            
            # Compute image embeddings
            inputs = {"pixel_values": patched}
            image_embs = self.model.get_image_features(**inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            
            # Compute similarity with target
            similarity = (image_embs @ self.target_emb.T).mean()
            
            # Loss: maximize similarity
            loss = -similarity + 0.01 * ((patch - patch.detach()).pow(2).mean())
            loss.backward()
            optimizer.step()
            
            # Clamp patch values
            with torch.no_grad():
                patch.clamp_(0, 1)
        
        return patch.detach()
    
    def apply_patch(self, images, patch, patch_location="random"):
        """
        Apply patch to images.
        
        Args:
            images: Image tensor or batch of images (B, C, H, W)
            patch: Patch tensor (3, patch_size, patch_size)
            patch_location: "random" or "topleft"
            
        Returns:
            Patched images tensor
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        patched = images.clone()
        patch_size = patch.shape[-1]
        
        for i in range(patched.size(0)):
            if patch_location == "random":
                x = random.randint(0, patched.shape[-1] - patch_size)
                y = random.randint(0, patched.shape[-2] - patch_size)
            else:
                x = y = 0
            patched[i, :, y:y+patch_size, x:x+patch_size] = patch
        
        return patched

