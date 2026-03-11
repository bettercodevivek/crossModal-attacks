"""
Projected Gradient Descent (PGD) Attack for CLIP.

This attack performs iterative gradient ascent with projection to maximize
CLIP similarity with a target caption.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import get_device, clamp_tensor


class PGDAttack:
    """
    PGD adversarial attack.
    
    Iterative algorithm:
    for step in range(num_steps):
        x_adv = x_adv + alpha * sign(grad(loss))
        x_adv = clip(x_adv, x-epsilon, x+epsilon)
    """
    
    def __init__(self, model, processor, target_text, device=None):
        """
        Initialize PGD attack.
        
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
    
    def attack(self, images, epsilon=None, num_steps=None, alpha=None):
        """
        Generate PGD adversarial examples.
        
        Args:
            images: Image tensor or batch of images (B, C, H, W)
            epsilon: Perturbation budget (default from Config)
            num_steps: Number of PGD steps (default from Config)
            alpha: Step size (default from Config)
            
        Returns:
            Adversarial images tensor
        """
        epsilon = epsilon or Config.PGD_EPSILON
        num_steps = num_steps or Config.PGD_STEPS
        alpha = alpha or Config.PGD_ALPHA
        
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        images = images.to(self.device)
        original_images = images.clone()
        
        # Initialize adversarial examples (start from original)
        adversarial = images.clone()
        adversarial.requires_grad = True
        
        # PGD iterations
        for step in range(num_steps):
            self.model.zero_grad()
            
            if adversarial.grad is not None:
                adversarial.grad.zero_()
            
            # Forward pass
            inputs = {"pixel_values": adversarial}
            image_embs = self.model.get_image_features(**inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            
            # Compute similarity with target
            similarity = (image_embs @ self.target_emb.T).mean()
            
            # Backward pass
            loss = -similarity  # Maximize similarity
            loss.backward()
            
            # Update: x_adv = x_adv + alpha * sign(grad)
            with torch.no_grad():
                grad_sign = adversarial.grad.sign()
                adversarial = adversarial + alpha * grad_sign
                
                # Project back to epsilon-ball: clip(x_adv, x-epsilon, x+epsilon)
                adversarial = torch.clamp(
                    adversarial, 
                    original_images - epsilon, 
                    original_images + epsilon
                )
                
                # Clip to valid image range [0, 1]
                adversarial = clamp_tensor(adversarial, 0.0, 1.0)
                
                # Re-enable gradients for next iteration
                adversarial.requires_grad = True
        
        return adversarial.detach()

