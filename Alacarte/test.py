import os
import numpy as np
import torch

from Alacarte.optimal_pattern import AlacartePattern

alacarte = AlacartePattern(
    pat_width=800, pat_height=600, n_patterns=4, mu=20,
    cam_width=800, tolerance=0, geom_constraints=None,
    output_dir="./Alacarte/testimg",
    device='cuda'
)
alacarte.optimize(n_iters=10000, logdir="./Alacarte/testlog")
# alacarte.optimize(logdir="./Alacarte/testlog")
alacarte.save()