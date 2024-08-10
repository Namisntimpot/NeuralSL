import os
import numpy as np
import torch

from Alacarte.optimal_pattern import AlacartePattern

alacarte = AlacartePattern(
    pat_width=800, pat_height=600, n_patterns=5, mu=300, ambient_max=0.1,
    cam_width=800, tolerance=0, geom_constraints=None,
    output_dir="./Alacarte/testimg/active/patterns", maxF=4, n_samples_for_eval=250,
    device='cuda'
)
alacarte.optimize(n_iters=1000, logdir="./Alacarte/testlog")
# alacarte.optimize(logdir="./Alacarte/testlog")
alacarte.save()