"""
Adversarial attack modules for cross-modal attacks on CLIP.
"""
from .patch_attack import PatchAttack
from .fgsm_attack import FGSMAttack
from .pgd_attack import PGDAttack

__all__ = ['PatchAttack', 'FGSMAttack', 'PGDAttack']

