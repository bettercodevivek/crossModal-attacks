"""
Configuration parameters for cross-modal adversarial attacks.
"""
import os

class Config:
    # Data directories
    DATA_DIR = "data/images"
    HOLDOUT_DIR = "data/holdout"
    OUTPUT_DIR = "output"
    RESULTS_DIR = "results"
    RESULTS_IMAGES_DIR = os.path.join("results", "images")
    
    # Model settings
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224
    
    # Patch attack parameters
    PATCH_SIZE = 100
    PATCH_STEPS = 800
    PATCH_BATCH_SIZE = 8
    PATCH_LR = 0.1
    PATCH_LOCATION = "random"  # "topleft" or "random"
    
    # FGSM attack parameters
    FGSM_EPSILON = 0.03
    
    # PGD attack parameters
    PGD_EPSILON = 0.03
    PGD_STEPS = 40
    PGD_ALPHA = 0.01
    
    # Target caption
    TARGET_TEXT = "a photo of a banana"
    
    # Candidate captions for evaluation
    CANDIDATE_TEXTS = [
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a person",
        "a photo of a car",
        "a photo of a banana"
    ]
    
    # Evaluation settings
    BATCH_SIZE = 8
    
    # Ensure output directories exist
    @staticmethod
    def ensure_dirs():
        """Create output directories if they don't exist."""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_IMAGES_DIR, exist_ok=True)
