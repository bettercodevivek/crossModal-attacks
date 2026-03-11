"""
Robustness evaluation pipeline for cross-modal adversarial attacks.
"""
import os
import json
import torch
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import get_device, load_image_tensors
from evaluation.metrics import compute_all_metrics


class RobustnessEvaluator:
    """
    Robustness evaluation pipeline.
    
    Steps:
    1) Load images from dataset
    2) Run selected attack
    3) Evaluate metrics
    4) Save results
    """
    
    def __init__(self, model, processor, device=None):
        """
        Initialize robustness evaluator.
        
        Args:
            model: CLIP model
            processor: CLIP processor
            device: Device to run on (auto-detected if None)
        """
        self.model = model
        self.processor = processor
        self.device = device if device else get_device()
    
    def evaluate_attack(self, attack, image_paths, target_text, batch_size=None, 
                       save_results=True, results_dir=None):
        """
        Evaluate an attack on a set of images.
        
        Args:
            attack: Attack object (PatchAttack, FGSMAttack, or PGDAttack)
            image_paths: List of image file paths
            target_text: Target caption
            batch_size: Batch size for evaluation
            save_results: Whether to save results to JSON
            results_dir: Directory to save results (default from Config)
            
        Returns:
            Dictionary with evaluation results
        """
        batch_size = batch_size or Config.BATCH_SIZE
        results_dir = results_dir or Config.RESULTS_DIR
        
        if len(image_paths) == 0:
            return {
                'attack': attack.__class__.__name__,
                'asr': 0.0,
                'avg_confidence_shift': 0.0,
                'robustness_score': 1.0,
                'num_images': 0
            }
        
        # Load images
        print(f"Loading {len(image_paths)} images...")
        image_tensors = load_image_tensors(image_paths, self.device)
        
        # Prepare for batch processing
        all_original = []
        all_adversarial = []
        
        # Process in batches
        print("Generating adversarial examples...")
        for i in tqdm(range(0, len(image_tensors), batch_size)):
            batch_indices = range(i, min(i + batch_size, len(image_tensors)))
            batch_tensors = [image_tensors[j] for j in batch_indices]
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Generate adversarial examples
            if isinstance(attack, type) and attack.__name__ == 'PatchAttack':
                # For patch attack, we need to apply the pre-generated patch
                # This assumes patch was already generated
                raise NotImplementedError("Patch attack evaluation requires pre-generated patch")
            else:
                adversarial_batch = attack.attack(batch)
            
            all_original.append(batch.cpu())
            all_adversarial.append(adversarial_batch.cpu())
        
        # Concatenate all batches
        original_images = torch.cat(all_original, dim=0)
        adversarial_images = torch.cat(all_adversarial, dim=0)
        
        # Compute metrics
        print("Computing metrics...")
        metrics = compute_all_metrics(
            self.model,
            self.processor,
            original_images.to(self.device),
            adversarial_images.to(self.device),
            target_text,
            self.device
        )
        
        # Prepare results
        results = {
            'attack': attack.__class__.__name__,
            'asr': metrics['asr'],
            'avg_confidence_shift': metrics['avg_confidence_shift'],
            'robustness_score': metrics['robustness_score'],
            'num_images': len(image_paths)
        }
        
        # Save results
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, 'metrics.json')
            
            # Load existing results if any
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    all_results = json.load(f)
            else:
                all_results = []
            
            # Append new results
            all_results.append(results)
            
            # Save
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\nResults saved to {results_file}")
        
        return results, original_images, adversarial_images

