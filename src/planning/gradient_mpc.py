# /workspace/src/planning/gradient_mpc.py

import torch
import warp as wp
from tqdm import trange
import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv
from src.planning.losses import chamfer, mean_chamfer_torch

def simulate_trajectory(env, action_seq, init_state_path, max_delta=0.03):
    """
    Simulate the trajectory in the environment with clamped actions.
    Returns predicted control and GS trajectories.
    """
    # 注释掉这一行，避免每一步都 set init state
    # env.set_init_state_from_numpy(init_state_path)

    pred_ctrl_traj = []
    pred_gs_traj = []

    for t in range(len(action_seq)):
        action_step = torch.clamp(action_seq[t], min=-max_delta, max=max_delta)

        env.step(env.n_ctrl_parts, action_step)

        obs = env.get_obs()
        # ctrl = torch.from_numpy(obs["ctrl_pts"]).float().to("cuda")
        # gs = torch.from_numpy(obs["state"]).float().to("cuda")
        ctrl = obs["ctrl_pts"]
        gs = obs["state"]

        pred_ctrl_traj.append(ctrl)
        pred_gs_traj.append(gs)

        # print(f"[Step {t}] action_step.requires_grad = {action_step.requires_grad}")
        # print(f"[Step {t}] ctrl.requires_grad = {ctrl.requires_grad}, gs.requires_grad = {gs.requires_grad}")

    pred_ctrl_traj = torch.stack(pred_ctrl_traj)  # (H, N, 3)
    pred_gs_traj = torch.stack(pred_gs_traj)      # (H, M, 3)
    
    return pred_ctrl_traj, pred_gs_traj


def compute_loss(pred_ctrl_traj, pred_gs_traj, target_ctrl_pts, target_gs_pts, action_seq, 
                 smooth_weight=0.1, use_mean_chamfer=False):
    """
    Compute total loss from Chamfer + Ctrl MSE + Smoothness.
    Optionally use mean chamfer.
    """
    # Chamfer loss (end GS state)
    chamfer_loss = chamfer(pred_gs_traj[-1:], target_gs_pts.unsqueeze(0)).mean()

    if use_mean_chamfer:
        state_pred = pred_gs_traj[-1]         # (M_pred, 3)
        state_real = target_gs_pts            # (M_target, 3)
        state_pred_mask = torch.ones(len(state_pred), dtype=torch.bool, device="cuda")
        state_real_mask = torch.ones(len(state_real), dtype=torch.bool, device="cuda")
        mean_chamfer_loss = mean_chamfer_torch(state_pred, state_real, state_pred_mask, state_real_mask)
        chamfer_loss = mean_chamfer_loss

    # Controller loss (final frame)
    ctrl_loss = torch.nn.functional.mse_loss(pred_ctrl_traj[-1], target_ctrl_pts)

    # Smoothness loss: ||a_t+1 - a_t||^2
    smooth_loss = torch.mean((action_seq[1:] - action_seq[:-1]) ** 2)

    total_loss = chamfer_loss + ctrl_loss + smooth_weight * smooth_loss

    return total_loss, chamfer_loss, ctrl_loss, smooth_loss


def run_gradient_mpc(env, target_gs_pts, target_ctrl_pts, init_state_path, horizon=40, lr=1e-2, outer_iters=200):
    def print_grad_hook(name):
        def hook(grad):
            print(f"🔍 Grad for {name}: norm = {grad.norm():.6f}")
            return grad
        return hook

    action_seq = torch.zeros((horizon, env.n_ctrl_parts, 3), requires_grad=True, device="cuda")
    action_seq.register_hook(print_grad_hook("action_seq"))

    optimizer = torch.optim.Adam([action_seq], lr=lr)

    # 只 set_init_state_from_numpy 一次
    env.set_init_state_from_numpy(init_state_path)

    for outer in range(outer_iters):
        pred_ctrl_traj, pred_gs_traj = simulate_trajectory(env, action_seq, init_state_path=init_state_path)

        loss, chamfer_loss, ctrl_loss, smooth_loss = compute_loss(
            pred_ctrl_traj, pred_gs_traj, target_ctrl_pts, target_gs_pts, action_seq, 
            smooth_weight=0.1, use_mean_chamfer=False
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print diagnostics
        grad_norm = action_seq.grad.norm().item() if action_seq.grad is not None else 0.0
        print(f"[iter {outer}] total loss={loss.detach().item():.4f} | chamfer={chamfer_loss.item():.4f} | ctrl={ctrl_loss.item():.4f} | smooth={smooth_loss.item():.4f} | grad_norm={grad_norm:.4f}")


env = PhysTwinEnv(case_name="double_lift_cloth_1")
init_state_path = os.path.join("PhysTwin", "mpc_init", "init_002.pkl")
target_state_path = os.path.join("PhysTwin", "mpc_target_U", "target_002.pkl")

target_state = pickle.load(open(target_state_path, "rb"))
target_gs_pts = torch.tensor(target_state["gs_pts"], dtype=torch.float32, device="cuda")
target_ctrl_pts = torch.tensor(target_state["ctrl_pts"], dtype=torch.float32, device="cuda")

run_gradient_mpc(env, target_gs_pts, target_ctrl_pts, init_state_path=init_state_path, horizon=40, lr=5e-2, outer_iters=300)

