"""
Full reproducibility helpers - both train.py and train_stageB.py previously
only called torch.manual_seed() and np.random.seed(), which leaves real
gaps: CUDA's own RNG state is separate from torch.manual_seed() and needs
torch.cuda.manual_seed_all() explicitly; cuDNN's default algorithm
selection is non-deterministic even with every RNG seeded, unless
cudnn.deterministic is set; and KLAPairDataset/ExternalImageDataset's
augmentation calls np.random.rand()/randint() inside __getitem__, which
runs in separate DataLoader worker processes when num_workers>0 - seeding
the main process's numpy RNG before creating the DataLoader does NOT
propagate to those workers without an explicit worker_init_fn + generator,
per PyTorch's own documented reproducibility guidance.
"""
import random

import numpy as np
import torch


def set_full_determinism(seed: int):
    """Call once at the start of main(), before any model/data construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no CUDA device, safe to always call
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Pass as DataLoader(..., worker_init_fn=seed_worker, generator=<seeded generator>)
    so each worker process's numpy/random state is deterministic - without
    this, augmentation calls inside __getitem__ (which run in the worker
    process, not the main one) are not reproducibly seeded when
    num_workers>0."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
