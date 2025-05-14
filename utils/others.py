import random
import numpy as np
import torch


def set_all_seeds(seed=42):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

