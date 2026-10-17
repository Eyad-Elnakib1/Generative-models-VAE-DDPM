import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _fixed_seed():
    """Make every test deterministic, so a statistical tolerance cannot fail by chance on some runs."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
