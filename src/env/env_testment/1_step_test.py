# /workspace/src/env/env_testment/1_step_test.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv
import numpy as np

env = PhysTwinEnv(case_name="double_lift_cloth_1")
obs = env.reset()

print("✅ Initial obs:", {k: v.shape for k, v in obs.items()})

for _ in range(5):
    delta = np.zeros((30, 3))  # no movement
    obs = env.step(delta)
    env.render()
