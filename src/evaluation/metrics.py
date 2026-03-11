"""
Evaluation metrics for cross-modal adversarial attacks.
"""
import torch
import numpy as np


def compute_similarity(model, processor, images, target_text, device="cpu"):
    """
    Compute CLIP similarity between images and target text.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        images: Image tensor (B, C, H, W)
        target_text: Target caption string
        device: Device to run on
        
    Returns:
        Similarity scores (B,)
    """
    with torch.no_grad():
        # Get image embeddings
        inputs = {"pixel_values": images}
        image_embs = model.get_image_features(**inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        
        # Get target text embedding
        text_inputs = processor(
            text=[target_text], 
            return_tensors="pt", 
            padding=True
        ).to(device)
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (image_embs @ text_emb.T).squeeze(-1)
        
    return similarity.cpu().numpy()


def compute_asr(original_similarities, adversarial_similarities):
    """
    Compute Attack Success Rate (ASR).
    
    ASR = successful_attacks / total_images
    
    A successful attack occurs when:
    similarity(target_caption, attacked_image) > similarity(target_caption, original_image)
    
    Args:
        original_similarities: Similarity scores for original images (N,)
        adversarial_similarities: Similarity scores for adversarial images (N,)
        
    Returns:
        Attack Success Rate (float between 0 and 1)
    """
    if len(original_similarities) == 0:
        return 0.0
    
    successful = adversarial_similarities > original_similarities
    asr = successful.sum() / len(successful)
    return float(asr)


def compute_confidence_shift(original_similarities, adversarial_similarities):
    """
    Compute confidence shift.
    
    confidence_shift = adversarial_similarity - original_similarity
    
    Args:
        original_similarities: Similarity scores for original images (N,)
        adversarial_similarities: Similarity scores for adversarial images (N,)
        
    Returns:
        Average confidence shift (float)
    """
    if len(original_similarities) == 0:
        return 0.0
    
    shifts = adversarial_similarities - original_similarities
    return float(shifts.mean())


def compute_robustness_score(asr):
    """
    Compute robustness score.
    
    R = 1 - ASR
    
    Args:
        asr: Attack Success Rate (float between 0 and 1)
        
    Returns:
        Robustness score (float between 0 and 1)
    """
    return 1.0 - asr


def compute_all_metrics(model, processor, original_images, adversarial_images, 
                       target_text, device="cpu"):
    """
    Compute all evaluation metrics.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        original_images: Original image tensors (B, C, H, W)
        adversarial_images: Adversarial image tensors (B, C, H, W)
        target_text: Target caption string
        device: Device to run on
        
    Returns:
        Dictionary with metrics:
        {
            'asr': float,
            'avg_confidence_shift': float,
            'robustness_score': float,
            'original_similarities': np.array,
            'adversarial_similarities': np.array
        }
    """
    # Compute similarities
    orig_sims = compute_similarity(model, processor, original_images, target_text, device)
    adv_sims = compute_similarity(model, processor, adversarial_images, target_text, device)
    
    # Compute metrics
    asr = compute_asr(orig_sims, adv_sims)
    conf_shift = compute_confidence_shift(orig_sims, adv_sims)
    robustness = compute_robustness_score(asr)
    
    return {
        'asr': asr,
        'avg_confidence_shift': conf_shift,
        'robustness_score': robustness,
        'original_similarities': orig_sims.tolist(),
        'adversarial_similarities': adv_sims.tolist()
    }

