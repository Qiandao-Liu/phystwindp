# run_mpc.py
import os
import torch
import numpy as np
import warp as wp
import pickle
from tqdm import tqdm
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg
from sklearn.cluster import KMeans

def load_trainer(case_name, base_path="./data/different_types"):
    cfg.load_from_yaml("configs/cloth.yaml")
    data_path = os.path.join(base_path, case_name, "final_data.pkl")
    base_dir = os.path.join("./temp_mpc_output", case_name)
    os.makedirs(base_dir, exist_ok=True)
    return InvPhyTrainerWarp(data_path=data_path, base_dir=base_dir, pure_inference_mode=True)

def mpc_optimize(trainer, idxs_left, idxs_right, n_steps=40, n_iters=300, lr=5e-2):
    ctrl_init = trainer.simulator.controller_points[0].detach().clone()
    ctrl_mask = torch.zeros(ctrl_init.shape[0], dtype=torch.bool, device=ctrl_init.device)
    ctrl_mask[idxs_left] = True
    ctrl_mask[idxs_right] = True
    num_ctrl = ctrl_mask.sum().item()

    offsets = [torch.nn.Parameter(torch.zeros(num_ctrl, 3, device=ctrl_init.device)) for _ in range(n_steps)]
    optimizer = torch.optim.Adam(offsets, lr=lr)

    # 目标位移（例如左右角对折）
    displacement = torch.zeros_like(ctrl_init)
    displacement[idxs_left] = torch.tensor([0.2, 1.0, 0.8], device=ctrl_init.device)
    displacement[idxs_right] = torch.tensor([0.2, 2.0, 0.5], device=ctrl_init.device)
    target = ctrl_init + displacement

    for iter in tqdm(range(n_iters), desc="MPC Optimization"):
        trainer.simulator.set_init_state(
            trainer.simulator.wp_init_vertices,
            trainer.simulator.wp_init_velocities,
            pure_inference=True
        )
        prev_ctrl = ctrl_init.clone()
        ctrl_trajectory = []

        for t in range(n_steps):
            next_ctrl = prev_ctrl.clone()
            next_ctrl[ctrl_mask] += offsets[t]

            trainer.simulator.set_controller_interactive(prev_ctrl.detach(), next_ctrl.detach())

            if trainer.simulator.object_collision_flag:
                trainer.simulator.update_collision_graph()
            trainer.simulator.step()
            trainer.simulator.set_init_state(
                trainer.simulator.wp_states[-1].wp_x,
                trainer.simulator.wp_states[-1].wp_v,
                pure_inference=True
            )

            prev_ctrl = next_ctrl
            ctrl_trajectory.append(next_ctrl[ctrl_mask])

        traj = torch.stack(ctrl_trajectory, dim=0)
        loss = torch.mean((traj - target[ctrl_mask]) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 20 == 0:
            print(f"Iter {iter} | Loss: {loss.item():.6f}")
    return offsets, ctrl_mask

def save_trajectory(trainer, offsets, ctrl_mask, filename="mpc_trajectory.pkl"):
    ctrl_init = trainer.simulator.controller_points[0].detach().clone()
    full_traj = []
    ctrl_traj = []

    for t in range(len(offsets)):
        ctrl_now = ctrl_init.clone()
        ctrl_now[ctrl_mask] += offsets[t].detach()

        trainer.simulator.set_controller_interactive(
            trainer.simulator.controller_points[0],
            ctrl_now
        )

        if trainer.simulator.object_collision_flag:
            trainer.simulator.update_collision_graph()

        wp.capture_launch(trainer.simulator.forward_graph)

        x = wp.to_torch(trainer.simulator.wp_states[-1].wp_x, requires_grad=False).detach().cpu().numpy()
        full_traj.append(x)
        ctrl_traj.append(ctrl_now.detach().cpu().numpy())

        trainer.simulator.set_init_state(
            trainer.simulator.wp_states[-1].wp_x,
            trainer.simulator.wp_states[-1].wp_v,
            pure_inference=True
        )

    with open(filename, "wb") as f:
        pickle.dump({
            "traj": np.stack(full_traj, axis=0),        # [T, N_all, 3]
            "ctrl_traj": np.stack(ctrl_traj, axis=0),   # [T, 30, 3]
        }, f)
    print(f"✅ Saved full particle trajectory to {filename}")


if __name__ == "__main__":
    case_name = "double_lift_cloth_3"
    trainer = load_trainer(case_name)
    ctrl_pts = trainer.simulator.controller_points[0]

    ctrl_pts_np = ctrl_pts.cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(ctrl_pts_np)
    labels = kmeans.labels_

    idxs_left = torch.where(torch.tensor(labels == 0))[0].to(ctrl_pts.device)
    idxs_right = torch.where(torch.tensor(labels == 1))[0].to(ctrl_pts.device)

    offsets, ctrl_mask = mpc_optimize(trainer, idxs_left, idxs_right)
    save_trajectory(trainer, offsets, ctrl_mask, f"./temp_mpc_output/{case_name}/mpc_trajectory.pkl")
