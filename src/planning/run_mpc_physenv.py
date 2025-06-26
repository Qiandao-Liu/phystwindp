# /workspace/src/planning/run_mpc_physenv.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))
from src.env.phystwin_env import PhysTwinEnv
from src.planning.losses import chamfer
from src.planning.mppi import MPPI
import numpy as np
import torch.nn.functional as F
import torch
import pickle
import wandb
import warnings
warnings.filterwarnings("ignore")
import builtins

original_print = builtins.print

def filtered_print(*args, **kwargs):
    if len(args) == 0:
        return  # skip empty print()
    # Join all args into a single string
    msg = " ".join(str(a) for a in args).strip()
    if msg == "":
        return  # skip print("") or print("\n")
    original_print(*args, **kwargs)

builtins.print = filtered_print

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--init_idx', type=int, default=0)
parser.add_argument('--target_idx', type=int, default=0)
parser.add_argument('--outer_iters', type=int, default=5)
parser.add_argument('--horizon', type=int, default=50)
args = parser.parse_args()

def run_mpc(init_idx=0, target_idx=0):
    wandb.init(
        project="phystwin-mpc",
        name=f"init{init_idx:03d}_target{target_idx:03d}",
        config={
            "init_idx": init_idx,
            "target_idx": target_idx,
            "horizon":50,
            "controller": "MPPI",
            "env": "PhysTwin",
        }
    )

    env = PhysTwinEnv()

    # ===== 1. Load full init state from .pkl =====
    init_path = f"PhysTwin/mpc_init/init_{init_idx:03d}.pkl"
    print(f"🔄 Loading init state from {init_path}")
    env.set_init_state_from_numpy(init_path)

    # ===== 2. Load target state (ctrl_pts + gs_pts only) =====
    target_path = f"PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"
    print(f"🎯 Loading target state from {target_path}")
    with open(target_path, "rb") as f:
        target_data = pickle.load(f)
    target_ctrl = torch.tensor(target_data["ctrl_pts"], dtype=torch.float32).cuda()
    target_gs = torch.tensor(target_data["gs_pts"], dtype=torch.float32).cuda()

    # ===== 3. MPPI Optimization =====
    mppi = MPPI(env, horizon=50, num_samples=512, lambda_=1.0, noise_sigma=0.05, outer_iters=args.outer_iters, init_idx=init_idx, target_idx=target_idx, init_path=init_path)
    optimized_actions = mppi.optimize(target_ctrl, target_gs)  # shape (H, 30, 3)

    # ===== 4. Re-execute optimized traj to record actual result =====
    env.set_init_state_from_numpy(init_path)

    ctrl_traj, gs_traj = [], []
    for t in range(optimized_actions.shape[0]):
        env.step(env.n_ctrl_parts, optimized_actions[t])
        ctrl_traj.append(env.get_obs()["ctrl_pts"])
        gs_traj.append(env.get_obs()["state"])

    # ===== 5. Compute final loss =====
    final_ctrl = torch.tensor(ctrl_traj[-1], dtype=torch.float32).cuda()
    final_gs = torch.tensor(gs_traj[-1], dtype=torch.float32).cuda()
    chamfer_loss = chamfer(final_gs[None], target_gs[None])[0]
    ctrl_loss = F.mse_loss(final_ctrl, target_ctrl)
    print(f"✅ Final Loss: Chamfer={chamfer_loss.item():.6f}, Ctrl MSE={ctrl_loss.item():.6f}")
    wandb.log({
        "final_chamfer_loss": chamfer_loss.item(),
        "final_ctrl_mse": ctrl_loss.item()
    })

    # ===== 6. Save optimized trajectory =====
    save_path = f"PhysTwin/mpc_output/init{init_idx:03d}_target{target_idx:03d}_iters{args.outer_iters}.pkl"

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
    wandb.finish()


if __name__ == "__main__":
    run_mpc(init_idx=0, target_idx=0)
