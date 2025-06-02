# workspace/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/deformable/deformable_cloth_env.py

import os
import sys
import gym
import numpy as np

# 确保能 import PhysTwin
PHYSTWIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../PhysTwin"))
sys.path.append(PHYSTWIN_ROOT)

from qqtt.engine.dp_warp import DPClothWarp

class DeformableClothEnv(gym.Env):
    def __init__(self, case_name="double_lift_cloth_1"):
        self.simulator = DPClothWarp(case_name=case_name)
        self.max_steps = self.simulator.max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        obs = self.simulator.reset()
        return obs

    def step(self, action):
        self.step_count += 1
        return self.simulator.step(action)
