# /workspace/src/planning/run_mpc_physenv.py
from src.env.phystwin_env import PhysTwinEnv
from src.planning.losses import chamfer
from src.planning.mppi import MPPI
import numpy as np
import torch.nn.functional as F
import torch
import os
import pickle

def load_target_state(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['ctrl_pts'], data['gs_pts']

def run_mpc(init_idx, target_idx):
    env = PhysTwinEnv()
    
    # ===== 1. Reset to initial state =====
    init_path = f"/workspace/PhysTwin/mpc_init/target_{init_idx:03d}.pkl"
    init_ctrl, init_gs = load_target_state(init_path)
    env.reset_to_origin()
    env.set_init_state_from_numpy(init_gs, init_ctrl)

    # ===== 2. Load target state =====
    target_path = f"/workspace/PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"
    target_ctrl, target_gs = load_target_state(target_path)
    target_ctrl = torch.tensor(target_ctrl, dtype=torch.float32).cuda()
    target_gs = torch.tensor(target_gs, dtype=torch.float32).cuda()

    # ===== 3. MPPI Optimization =====
    mppi = MPPI(env)
    optimized_actions = mppi.optimize(target_ctrl, target_gs)  # shape (H, 30, 3)

    # ===== 4. Re-execute optimized traj to record actual result =====
    env.reset_to_origin()
    env.set_init_state_from_numpy(init_gs, init_ctrl)

    ctrl_traj, gs_traj = [], []
    for t in range(optimized_actions.shape[0]):
        env.step(optimized_actions[t])
        ctrl_traj.append(env.get_ctrl_pts())
        gs_traj.append(env.get_gs_pts())

    # ===== 5. Compute final loss =====
    final_ctrl = torch.tensor(ctrl_traj[-1], dtype=torch.float32).cuda()
    final_gs = torch.tensor(gs_traj[-1], dtype=torch.float32).cuda()
    chamfer_loss = chamfer(final_gs[None], target_gs[None])[0]
    ctrl_loss = F.mse_loss(final_ctrl, target_ctrl)
    print(f"🎯 Final Loss: Chamfer={chamfer_loss.item():.6f}, Ctrl MSE={ctrl_loss.item():.6f}")

    # ===== 6. Save optimized trajectory =====
    save_path = f"/workspace/PhysTwin/mpc_output/rollout_init{init_idx:03d}_target{target_idx:03d}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "ctrl_traj": np.array(ctrl_traj),     # (H, 30, 3)
            "gs_traj": np.array(gs_traj),         # (H, N, 3)
            "target_ctrl": target_ctrl.detach().cpu().numpy(),
            "target_gs": target_gs.detach().cpu().numpy(),
            "optimized_actions": optimized_actions.detach().cpu().numpy(),
        }, f)
    print(f"💾 Saved trajectory to {save_path}")


if __name__ == "__main__":
    run_mpc(init_idx=0, target_idx=0)
