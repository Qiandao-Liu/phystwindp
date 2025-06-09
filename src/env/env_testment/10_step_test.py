# /workspace/src/env/env_testment/10_step_test.py
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv
import numpy as np

env = PhysTwinEnv(case_name="double_lift_cloth_1")
obs = env.reset()

print("✅ Initial obs:", {k: v.shape for k, v in obs.items()})

for i in range(10):
    delta = torch.randn(30, 3) * 0.001  # small random move
    obs = env.step(delta)
    env.render()

    # Check stats
    ctrl_pos = obs["ctrl_pts"]
    obj_pos = obs["gs_pts"]
    print(f"   Mean ctrl pos: {ctrl_pos.mean().item():.4f} | Mean obj pos: {obj_pos.mean().item():.4f}")
