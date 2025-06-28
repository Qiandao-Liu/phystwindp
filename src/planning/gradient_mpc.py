# /workspace/src/planning/gradient_mpc.py

import torch
import warp as wp
from warp.torch import from_torch
from tqdm import trange
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv
from src.planning.losses import chamfer

import sys
import faulthandler
faulthandler.enable()
import warnings
import os
# 禁用多次CUDA报错输出，只要报一次就停
os.environ["WARP_DISABLE_DEVICE_FREE_ASYNC_ERRORS"] = "1"
def error_handler(type, value, tb):
    print(f"❌ Fatal error: {value}")
    sys.exit(1)
sys.excepthook = error_handler


wp.config.enable_autodiff = True

# === 配置 ===
case_name = "double_lift_cloth_1"
init_idx = 0
target_idx = 0
H = 60                     # MPC steps
outer_iters = 200          # Optimization steps
save_steps = [1, 2, 5, 10, 20, 50, 100, 150, 200]

# === 加载环境 ===
env = PhysTwinEnv(case_name=case_name)

# === 加载起始和目标状态 ===
init_path = f"PhysTwin/mpc_init/init_{init_idx:03d}.pkl"
target_path = f"PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"

with open(target_path, "rb") as f:
    target_data = pickle.load(f)

env.set_init_state_from_numpy(init_path)

n_ctrl = env.n_ctrl_parts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化控制变量 ===
actions = torch.zeros((H, n_ctrl, 3), dtype=torch.float32, requires_grad=True, device=device)
optimizer = torch.optim.Adam([actions], lr=0.05)

# === 优化循环 ===
for step in trange(outer_iters):
    env.set_init_state_from_numpy(init_path)
    
    traj_ctrl, traj_gs = [], []
    
    tape = wp.Tape()
    with tape:
        for t in range(H):
            # action_numpy = actions[t].detach().cpu().numpy()
            # assert action_numpy.shape == (n_ctrl, 3), f"Expected shape {(n_ctrl,3)}, got {action_numpy.shape}"
            wp_action = from_torch(actions[t], dtype=wp.vec3, requires_grad=True)

            env.simulator.step_with_action(wp_action)
            traj_ctrl.append(env.get_ctrl_pts().detach().cpu().numpy())
            traj_gs.append(env.get_gs_pts())

        final_ctrl = torch.tensor(traj_ctrl[-1], device=device)
        final_gs = torch.tensor(traj_gs[-1], device=device)

        target_ctrl = torch.tensor(target_data["ctrl_pts"], device=device)
        target_gs = torch.tensor(target_data["gs_pts"], device=device)

        loss_ctrl = torch.nn.functional.mse_loss(final_ctrl, target_ctrl)
        loss_chamfer = chamfer(final_gs[None], target_gs[None])[0]
        loss = loss_ctrl + loss_chamfer

    optimizer.zero_grad()

    loss.backward()
    print(f"actions.grad.norm: {actions.grad.norm()}")
    
    optimizer.step()

    if step in save_steps:
        save_name = f"PhysTwin/mpc_output/grad_case={case_name}_init{init_idx:03d}_target{target_idx:03d}_step{step:03d}.pkl"
        with open(save_name, "wb") as f:
            pickle.dump({
                "ctrl_traj": np.array(traj_ctrl),
                "gs_traj": np.array(traj_gs),
                "optimized_actions": actions.detach().cpu().numpy(),
            }, f)
        print(f"📸 Saved intermediate traj at step {step} to {save_name}")

    if step % 20 == 0:
        print(f"[{step}] Loss: chamfer={loss_chamfer.item():.4f}, ctrl={loss_ctrl.item():.4f}")

# === 最终保存 ===
save_path = f"PhysTwin/mpc_output/grad_case={case_name}_init{init_idx:03d}_target{target_idx:03d}.pkl"
with open(save_path, "wb") as f:
    pickle.dump({
        "ctrl_traj": np.array(traj_ctrl),
        "gs_traj": np.array(traj_gs),
        "optimized_actions": actions.detach().cpu().numpy(),
        "target_ctrl": target_data["ctrl_pts"],
        "target_gs": target_data["gs_pts"],
    }, f)
print(f"✅ Saved gradient MPC trajectory to: {save_path}")
