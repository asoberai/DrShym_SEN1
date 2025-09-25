"""
Deterministic seed utilities for reproducible training
"""

import random
import numpy as np
import torch
import os


def set_deterministic_seed(seed: int = 42):
    """Set deterministic seed for reproducible training"""

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Set deterministic seed: {seed}")
    return seed


# Alias for compatibility
set_seed = set_deterministic_seed