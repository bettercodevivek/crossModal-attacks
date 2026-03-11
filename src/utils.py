"""
Utility functions for cross-modal adversarial attacks.
"""
import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms


def get_device():
    """Get the available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image_paths(folder):
    """
    Load all image file paths from a directory.
    
    Args:
        folder: Path to directory containing images
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(folder):
        return []
    
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files += glob(os.path.join(folder, e))
    return sorted(files)


def get_image_preprocessor():
    """
    Get the image preprocessing transform for CLIP.
    
    Returns:
        torchvision.transforms.Compose transform
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def open_and_preprocess(path, preprocess=None):
    """
    Open and preprocess an image.
    
    Args:
        path: Path to image file
        preprocess: Optional preprocessing transform
        
    Returns:
        Preprocessed image tensor
    """
    if preprocess is None:
        preprocess = get_image_preprocessor()
    
    img = Image.open(path).convert("RGB")
    return preprocess(img)


def load_image_tensors(image_paths, device="cpu"):
    """
    Load and preprocess multiple images.
    
    Args:
        image_paths: List of image file paths
        device: Device to load tensors on
        
    Returns:
        List of image tensors
    """
    preprocess = get_image_preprocessor()
    tensors = []
    for path in image_paths:
        tensor = open_and_preprocess(path, preprocess)
        tensors.append(tensor)
    return tensors


def tensor_to_image(tensor):
    """
    Convert a tensor to a numpy array image.
    
    Args:
        tensor: Image tensor (C, H, W) with values in [0, 1]
        
    Returns:
        numpy array (H, W, C) with values in [0, 255]
    """
    import numpy as np
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    return img


def clamp_tensor(tensor, min_val=0.0, max_val=1.0):
    """
    Clamp tensor values to a range.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped tensor
    """
    return torch.clamp(tensor, min_val, max_val)

