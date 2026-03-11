"""
Visualization functions for attack results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import tensor_to_image


def visualize_attack_results(original_images, adversarial_images, 
                            original_similarities, adversarial_similarities,
                            target_text, candidate_texts=None, 
                            save_dir=None, max_images=8):
    """
    Generate side-by-side comparison visualizations.
    
    Args:
        original_images: Original image tensors (B, C, H, W)
        adversarial_images: Adversarial image tensors (B, C, H, W)
        original_similarities: Similarity scores for original images
        adversarial_similarities: Similarity scores for adversarial images
        target_text: Target caption
        candidate_texts: Optional list of candidate texts for prediction display
        save_dir: Directory to save images (default from Config)
        max_images: Maximum number of images to visualize
    """
    save_dir = save_dir or Config.RESULTS_IMAGES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = min(max_images, len(original_images))
    
    for i in range(num_images):
        orig_img = tensor_to_image(original_images[i])
        adv_img = tensor_to_image(adversarial_images[i])
        
        orig_sim = original_similarities[i] if isinstance(original_similarities, (list, np.ndarray)) else original_similarities
        adv_sim = adversarial_similarities[i] if isinstance(adversarial_similarities, (list, np.ndarray)) else adversarial_similarities
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axs[0].imshow(orig_img)
        axs[0].set_title(
            f"Original\nSimilarity to '{target_text}': {orig_sim:.3f}",
            fontsize=10
        )
        axs[0].axis('off')
        
        # Adversarial image
        axs[1].imshow(adv_img)
        axs[1].set_title(
            f"Adversarial\nSimilarity to '{target_text}': {adv_sim:.3f}",
            fontsize=10
        )
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_{i:03d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {num_images} comparison images to {save_dir}")


def save_comparison_images(original_images, adversarial_images, 
                          original_similarities, adversarial_similarities,
                          target_text, save_dir=None, max_images=8):
    """
    Save comparison images (alias for visualize_attack_results).
    
    Args:
        original_images: Original image tensors (B, C, H, W)
        adversarial_images: Adversarial image tensors (B, C, H, W)
        original_similarities: Similarity scores for original images
        adversarial_similarities: Similarity scores for adversarial images
        target_text: Target caption
        save_dir: Directory to save images (default from Config)
        max_images: Maximum number of images to visualize
    """
    visualize_attack_results(
        original_images,
        adversarial_images,
        original_similarities,
        adversarial_similarities,
        target_text,
        save_dir=save_dir,
        max_images=max_images
    )

