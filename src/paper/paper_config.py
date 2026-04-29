"""
Hyperparameters aligned with the paper's methodology (Section V–VI).
"""
import os

# Paths (relative to project root when checkpoints saved from run script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "paper_checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "paper_results")


class PaperConfig:
    # Training
    BATCH_SIZE = 128
    EPOCHS_MNIST = 12
    EPOCHS_CIFAR = 25
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 42

    # Attacks (image tensors in [0, 1])
    FGSM_EPSILON = 0.03
    PGD_EPSILON = 0.03
    PGD_ALPHA = 0.01
    PGD_STEPS = 40

    # Carlini–Wagner L2 (margin + L2 trade-off)
    CW_STEPS = 100
    CW_LR = 0.01
    CW_C = 10.0

    # Adversarial training (PGD inner loop during train)
    ADV_TRAIN_EPS = 0.03
    ADV_TRAIN_ALPHA = 0.01
    ADV_TRAIN_STEPS = 7

    # Distillation
    DISTILL_TEMPERATURE = 5.0
    DISTILL_ALPHA = 0.7  # soft loss weight; (1-alpha) for hard CE

    @staticmethod
    def ensure_dirs():
        os.makedirs(PaperConfig.checkpoint_dir(), exist_ok=True)
        os.makedirs(PaperConfig.results_dir(), exist_ok=True)

    @staticmethod
    def checkpoint_dir():
        return CHECKPOINT_DIR

    @staticmethod
    def results_dir():
        return RESULTS_DIR
