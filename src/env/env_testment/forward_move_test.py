# /workspace/src/env/env_testment/forward_move_test.py
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

delta_dir = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(30, 1)  # (30, 3)
delta_mag = 0.005

delta = delta_dir * delta_mag

for i in range(200):
    obs = env.step(delta)

    if i % 10 == 0:
        env.render()

        # 控制点位置变化
        if i > 0:
            ctrl_diff = obs["ctrl_pts"] - prev_ctrl
            print(f"📍 Step {i:03d} | First ctrl point: {obs['ctrl_pts'][0].tolist()}")
        else:
            ctrl_diff = torch.zeros_like(obs["ctrl_pts"])

        # GS 点变化
        diff = obs["gs_pts"] - prev_gs if i > 0 else torch.zeros_like(obs["gs_pts"])
        print(f"Step {i:03d} | Δgs mean: {diff.norm(dim=1).mean():.6f}")
        print(f"📍 Step {i:03d} | First GS point: {obs['gs_pts'][0].tolist()}")

        prev_ctrl = obs["ctrl_pts"].clone()
        prev_gs = obs["gs_pts"].clone()
