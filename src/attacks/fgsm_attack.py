"""
Fast Gradient Sign Method (FGSM) Attack for CLIP.

This attack perturbs images using the sign of the gradient to maximize
CLIP similarity with a target caption.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import get_device, clamp_tensor


class FGSMAttack:
    """
    FGSM adversarial attack.
    
    Formula: x_adv = x + epsilon * sign(grad(loss))
    """
    
    def __init__(self, model, processor, target_text, device=None):
        """
        Initialize FGSM attack.
        
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
    
    def attack(self, images, epsilon=None):
        """
        Generate FGSM adversarial examples.
        
        Args:
            images: Image tensor or batch of images (B, C, H, W)
            epsilon: Perturbation budget (default from Config)
            
        Returns:
            Adversarial images tensor
        """
        epsilon = epsilon or Config.FGSM_EPSILON
        
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        images = images.to(self.device)
        images.requires_grad = True
        
        # Forward pass
        inputs = {"pixel_values": images}
        image_embs = self.model.get_image_features(**inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        
        # Compute similarity with target
        similarity = (image_embs @ self.target_emb.T).mean()
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss = -similarity  # Maximize similarity
        loss.backward()
        
        # FGSM: x_adv = x + epsilon * sign(grad)
        with torch.no_grad():
            grad_sign = images.grad.sign()
            adversarial = images + epsilon * grad_sign
            
            # Clip to valid image range [0, 1]
            adversarial = clamp_tensor(adversarial, 0.0, 1.0)
        
        return adversarial.detach()

