import os
import numpy as np
import torch

from Alacarte.optimal_pattern import AlacartePattern

alacarte = AlacartePattern(
    pat_width=800, pat_height=600, n_patterns=3, mu=10000, ambient_max=0.1,
    cam_width=800, tolerance=3, geom_constraints=None,
    output_dir="./Alacarte/testimg/active/patterns", maxF=128, n_samples_for_eval=250,
    device='cuda'
)  # 此处的samples_per_iter充当batch_size
alacarte.optimize(n_iters=6000, samples_per_iter=32, logdir="./Alacarte/testlog")
# alacarte.optimize(logdir="./Alacarte/testlog") 
alacarte.height = 600
alacarte.save()