"""
Main script for cross-modal adversarial attack robustness testing.

Usage:
    python demo_attack.py --attack patch
    python demo_attack.py --attack fgsm
    python demo_attack.py --attack pgd
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

from config import Config
from utils import get_device, load_image_paths, load_image_tensors
from attacks import PatchAttack, FGSMAttack, PGDAttack
from evaluation.robustness_evaluator import RobustnessEvaluator
from evaluation.metrics import compute_similarity
from visualization.visualize_results import visualize_attack_results


def load_clip_model(device=None):
    """
    Load CLIP model and processor.
    
    Args:
        device: Device to load on (auto-detected if None)
        
    Returns:
        model, processor tuple
    """
    device = device or get_device()
    print(f"Loading CLIP model on {device.upper()}...")
    model = CLIPModel.from_pretrained(Config.CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
    print("Model loaded successfully.\n")
    return model, processor


def run_patch_attack(model, processor, train_paths, eval_paths, device):
    """
    Run universal adversarial patch attack.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        train_paths: Training image paths
        eval_paths: Evaluation image paths
        device: Device to run on
        
    Returns:
        Results dictionary and images
    """
    print("\n" + "="*50)
    print("UNIVERSAL ADVERSARIAL PATCH ATTACK")
    print("="*50 + "\n")
    
    # Load training images
    print(f"Loading {len(train_paths)} training images...")
    train_tensors = load_image_tensors(train_paths, device)
    
    # Initialize patch attack
    patch_attack = PatchAttack(model, processor, Config.TARGET_TEXT, device)
    
    # Generate patch
    print(f"\nGenerating universal patch (this may take a few minutes)...")
    print(f"Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"Training steps: {Config.PATCH_STEPS}")
    print(f"Target caption: '{Config.TARGET_TEXT}'\n")
    
    patch = patch_attack.generate_patch(
        train_tensors,
        patch_size=Config.PATCH_SIZE,
        steps=Config.PATCH_STEPS,
        batch_size=Config.PATCH_BATCH_SIZE,
        lr=Config.PATCH_LR,
        patch_location=Config.PATCH_LOCATION
    )
    
    # Save patch
    patch_img = (patch.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    patch_path = os.path.join(Config.OUTPUT_DIR, "universal_patch.png")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    plt.imsave(patch_path, patch_img)
    print(f"Saved patch to: {patch_path}\n")
    
    # Evaluate on holdout images
    if len(eval_paths) == 0:
        print("No evaluation images found. Skipping evaluation.")
        return None, None, None
    
    print(f"Evaluating on {len(eval_paths)} holdout images...")
    eval_tensors = load_image_tensors(eval_paths, device)
    
    # Apply patch to evaluation images
    original_images = torch.stack(eval_tensors).to(device)
    adversarial_images = []
    
    for img in eval_tensors:
        img_batch = img.unsqueeze(0).to(device)
        patched = patch_attack.apply_patch(img_batch, patch, Config.PATCH_LOCATION)
        adversarial_images.append(patched.squeeze(0).cpu())
    
    adversarial_images = torch.stack(adversarial_images).to(device)
    
    # Compute metrics
    print("Computing metrics...")
    orig_sims = compute_similarity(model, processor, original_images, Config.TARGET_TEXT, device)
    adv_sims = compute_similarity(model, processor, adversarial_images, Config.TARGET_TEXT, device)
    
    # Calculate ASR
    successful = adv_sims > orig_sims
    asr = successful.sum() / len(successful) if len(successful) > 0 else 0.0
    conf_shift = float((adv_sims - orig_sims).mean())
    robustness = 1.0 - asr
    
    results = {
        'attack': 'PatchAttack',
        'asr': float(asr),
        'avg_confidence_shift': conf_shift,
        'robustness_score': robustness,
        'num_images': len(eval_paths)
    }
    
    return results, original_images.cpu(), adversarial_images.cpu()


def run_fgsm_attack(model, processor, eval_paths, device):
    """
    Run FGSM attack.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        eval_paths: Evaluation image paths
        device: Device to run on
        
    Returns:
        Results dictionary and images
    """
    print("\n" + "="*50)
    print("FGSM ATTACK")
    print("="*50 + "\n")
    
    if len(eval_paths) == 0:
        print("No evaluation images found.")
        return None, None, None
    
    # Initialize FGSM attack
    fgsm_attack = FGSMAttack(model, processor, Config.TARGET_TEXT, device)
    
    print(f"Running FGSM attack on {len(eval_paths)} images...")
    print(f"Epsilon: {Config.FGSM_EPSILON}")
    print(f"Target caption: '{Config.TARGET_TEXT}'\n")
    
    # Load and process images in batches
    eval_tensors = load_image_tensors(eval_paths, device)
    original_images = torch.stack(eval_tensors).to(device)
    
    # Generate adversarial examples
    adversarial_images = fgsm_attack.attack(original_images, Config.FGSM_EPSILON)
    
    # Compute metrics
    print("Computing metrics...")
    orig_sims = compute_similarity(model, processor, original_images, Config.TARGET_TEXT, device)
    adv_sims = compute_similarity(model, processor, adversarial_images, Config.TARGET_TEXT, device)
    
    # Calculate metrics
    successful = adv_sims > orig_sims
    asr = successful.sum() / len(successful) if len(successful) > 0 else 0.0
    conf_shift = float((adv_sims - orig_sims).mean())
    robustness = 1.0 - asr
    
    results = {
        'attack': 'FGSMAttack',
        'asr': float(asr),
        'avg_confidence_shift': conf_shift,
        'robustness_score': robustness,
        'num_images': len(eval_paths)
    }
    
    return results, original_images.cpu(), adversarial_images.cpu()


def run_pgd_attack(model, processor, eval_paths, device):
    """
    Run PGD attack.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        eval_paths: Evaluation image paths
        device: Device to run on
        
    Returns:
        Results dictionary and images
    """
    print("\n" + "="*50)
    print("PGD ATTACK")
    print("="*50 + "\n")
    
    if len(eval_paths) == 0:
        print("No evaluation images found.")
        return None, None, None
    
    # Initialize PGD attack
    pgd_attack = PGDAttack(model, processor, Config.TARGET_TEXT, device)
    
    print(f"Running PGD attack on {len(eval_paths)} images...")
    print(f"Epsilon: {Config.PGD_EPSILON}")
    print(f"Steps: {Config.PGD_STEPS}")
    print(f"Alpha: {Config.PGD_ALPHA}")
    print(f"Target caption: '{Config.TARGET_TEXT}'\n")
    
    # Load and process images in batches
    eval_tensors = load_image_tensors(eval_paths, device)
    original_images = torch.stack(eval_tensors).to(device)
    
    # Generate adversarial examples
    adversarial_images = pgd_attack.attack(
        original_images, 
        Config.PGD_EPSILON, 
        Config.PGD_STEPS, 
        Config.PGD_ALPHA
    )
    
    # Compute metrics
    print("Computing metrics...")
    orig_sims = compute_similarity(model, processor, original_images, Config.TARGET_TEXT, device)
    adv_sims = compute_similarity(model, processor, adversarial_images, Config.TARGET_TEXT, device)
    
    # Calculate metrics
    successful = adv_sims > orig_sims
    asr = successful.sum() / len(successful) if len(successful) > 0 else 0.0
    conf_shift = float((adv_sims - orig_sims).mean())
    robustness = 1.0 - asr
    
    results = {
        'attack': 'PGDAttack',
        'asr': float(asr),
        'avg_confidence_shift': conf_shift,
        'robustness_score': robustness,
        'num_images': len(eval_paths)
    }
    
    return results, original_images.cpu(), adversarial_images.cpu()


def save_results(results, results_dir=None):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        results_dir: Directory to save results (default from Config)
    """
    import json
    
    results_dir = results_dir or Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'metrics.json')
    
    # Load existing results if any
    all_results = []
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    
    # Append new results
    all_results.append(results)
    
    # Save
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Cross-Modal Adversarial Attack Robustness Testing'
    )
    parser.add_argument(
        '--attack',
        type=str,
        choices=['patch', 'fgsm', 'pgd'],
        required=True,
        help='Attack method to use'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default=None,
        help='Training image directory (required for patch attack)'
    )
    parser.add_argument(
        '--eval_dir',
        type=str,
        default=None,
        help='Evaluation image directory'
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Config.ensure_dirs()
    
    # Print header
    print("\n" + "="*50)
    print("CROSS-MODAL ADVERSARIAL ATTACK FRAMEWORK")
    print("="*50 + "\n")
    print("This framework tests robustness of CLIP to cross-modal attacks.")
    print("Attacks perturb images to manipulate text-based predictions.\n")
    
    # Get device
    device = get_device()
    print(f"Device: {device.upper()}\n")
    
    # Load model
    model, processor = load_clip_model(device)
    
    # Determine data directories
    train_dir = args.train_dir or Config.DATA_DIR
    eval_dir = args.eval_dir or Config.HOLDOUT_DIR
    
    # Load image paths
    train_paths = load_image_paths(train_dir)
    eval_paths = load_image_paths(eval_dir)
    
    if args.attack == 'patch':
        if len(train_paths) == 0:
            raise SystemExit(f"❌ ERROR: No training images found in {train_dir}")
        results, orig_imgs, adv_imgs = run_patch_attack(
            model, processor, train_paths, eval_paths, device
        )
    elif args.attack == 'fgsm':
        results, orig_imgs, adv_imgs = run_fgsm_attack(
            model, processor, eval_paths, device
        )
    elif args.attack == 'pgd':
        results, orig_imgs, adv_imgs = run_pgd_attack(
            model, processor, eval_paths, device
        )
    else:
        raise ValueError(f"Unknown attack: {args.attack}")
    
    if results is None:
        print("\nNo results to save.")
        return
    
    # Print results
    print("\n" + "="*50)
    print("ATTACK RESULTS")
    print("="*50 + "\n")
    print(f"Attack method: {results['attack']}")
    print(f"Attack Success Rate (ASR): {results['asr']*100:.2f}%")
    print(f"Average confidence shift: {results['avg_confidence_shift']:.4f}")
    print(f"Robustness score: {results['robustness_score']:.4f}")
    print(f"Number of images: {results['num_images']}\n")
    
    # Save results
    save_results(results)
    
    # Generate visualizations
    if orig_imgs is not None and adv_imgs is not None:
        print("\nGenerating visualizations...")
        orig_sims = compute_similarity(model, processor, orig_imgs.to(device), Config.TARGET_TEXT, device)
        adv_sims = compute_similarity(model, processor, adv_imgs.to(device), Config.TARGET_TEXT, device)
        
        visualize_attack_results(
            orig_imgs,
            adv_imgs,
            orig_sims,
            adv_sims,
            Config.TARGET_TEXT,
            save_dir=Config.RESULTS_IMAGES_DIR,
            max_images=8
        )
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
