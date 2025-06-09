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
delta_mag = 0.0005  # 每步微移 0.5mm

delta = delta_dir * delta_mag

# 运行 200 步，固定方向微移
for i in range(200):
    obs = env.step(delta)
    if i % 10 == 0:
        env.render()
        diff = obs["gs_pts"] - prev_gs if i > 0 else torch.zeros_like(obs["gs_pts"])
        print(f"🔍 Step {i:03d} | Δgs mean: {diff.norm(dim=1).mean():.6f}")
        prev_gs = obs["gs_pts"].clone()
