import os
import pickle
import torch
import numpy as np
import warp as wp
from tqdm import tqdm
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg

def load_trainer(case_name):
    cfg.load_from_yaml(os.path.expanduser("~/workspace/PhysTwin/configs/real.yaml"))
    base_dir = os.path.expanduser(f"~/workspace/PhysTwin/temp_mpc_output/{case_name}")
    data_path = os.path.expanduser(f"~/workspace/PhysTwin/data/different_types/{case_name}/final_data.pkl")
    return InvPhyTrainerWarp(data_path=data_path, base_dir=base_dir, pure_inference_mode=True)

def replay_trajectory(trainer, ctrl_traj):
    gs_traj = []

    # Set initial state only once at beginning
    trainer.set_init_state_from_numpy(
        full_traj[0], ctrl_traj[0]
    )


    for frame in ctrl_traj:
        ctrl = torch.tensor(frame, dtype=torch.float32, device=cfg.device)
        trainer.simulator.set_controller_interactive(
            trainer.simulator.controller_points[0], ctrl
        )
        trainer.simulator.step()

        trainer.simulator.set_init_state(
            trainer.simulator.wp_states[-1].wp_x,
            trainer.simulator.wp_states[-1].wp_v,
            pure_inference=True
        )

        x_wp = trainer.simulator.wp_states[-1].wp_x
        x_torch = wp.to_torch(x_wp, requires_grad=False)
        x_np = x_torch[:trainer.num_all_points].detach().cpu().numpy()

        gs_traj.append(x_np)

    return np.stack(gs_traj, axis=0)


if __name__ == "__main__":
    case_name = "rope_double_hand"
    traj_dir = os.path.expanduser(f"~/workspace/PhysTwin/temp_mpc_output/{case_name}")
    save_dir = os.path.expanduser(f"~/workspace/PhysTwin/replayed_trajs/{case_name}")
    os.makedirs(save_dir, exist_ok=True)

    trainer = load_trainer(case_name)

    traj_files = sorted([
        f for f in os.listdir(traj_dir)
        if f.startswith("mpc_trajectory_") and f.endswith(".pkl")
    ])

    for fname in tqdm(traj_files, desc="Replaying MPC Trajectories"):
        path = os.path.join(traj_dir, fname)
        with open(path, "rb") as f:
            data = pickle.load(f)

        ctrl_traj = data["ctrl_traj"]
        full_traj = data["traj"]
        gs_color = data["gs_color"]

        gs_traj = replay_trajectory(trainer, ctrl_traj)

        save_path = os.path.join(save_dir, fname.replace("trajectory", "replay"))
        with open(save_path, "wb") as f:
            pickle.dump({
                "ctrl_traj": ctrl_traj,
                "gs_traj": gs_traj,
                "gs_color": gs_color,
            }, f)
