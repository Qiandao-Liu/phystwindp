import os
import torch
import numpy as np
import warp as wp
import pickle
from tqdm import tqdm
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg
from sklearn.cluster import KMeans
from chamfer_distance import ChamferDistance
from os.path import expanduser
import open3d as o3d
import json
import glob

def load_trainer(case_name, base_path="~/workspace/PhysTwin/data/different_types"):
    cfg.load_from_yaml(expanduser("~/workspace/PhysTwin/configs/real.yaml"))
    load_optimal_params(case_name)
    data_path = os.path.join(expanduser(base_path), case_name, "final_data.pkl")
    base_dir = os.path.join(expanduser("~/workspace/PhysTwin/temp_mpc_output"), case_name)
    os.makedirs(base_dir, exist_ok=True)

    with open(os.path.join(expanduser(base_path), case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    with open(os.path.join(expanduser(base_path), case_name, "metadata.json"), "r") as f:
        metadata = json.load(f)
    intrinsics = np.array(metadata["intrinsics"])
    w2c = np.array(w2cs[0])
    intrinsic = intrinsics[0]

    trainer = InvPhyTrainerWarp(data_path=data_path, base_dir=base_dir, pure_inference_mode=True)
    trainer.intrinsic = intrinsic
    trainer.w2c = w2c

    trainer.simulator.dt = cfg.dt
    trainer.simulator.num_substeps = cfg.num_substeps

    return trainer

def load_optimal_params(case_name):
    path = expanduser(f"~/workspace/PhysTwin/experiments_optimization/{case_name}/optimal_params.pkl")
    with open(path, "rb") as f:
        cfg.set_optimal_params(pickle.load(f))

def load_gs_xyz(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points)
    return (
        torch.tensor(points, dtype=torch.float32, device=cfg.device),
        torch.tensor(colors, dtype=torch.float32, device=cfg.device),
    )

def load_ctrl_npy(npy_path):
    return torch.tensor(np.load(npy_path), dtype=torch.float32, device=cfg.device)

def get_hand_indices(ctrl_pts_np, intrinsic, w2c):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(ctrl_pts_np)
    labels = kmeans.labels_

    group_0 = ctrl_pts_np[labels == 0]
    group_1 = ctrl_pts_np[labels == 1]
    proj_mat = intrinsic @ w2c[:3, :]

    def get_projected_center(group):
        center = np.mean(group, axis=0)
        center = proj_mat @ np.append(center, 1)
        return center[:2] / center[2]

    center_0 = get_projected_center(group_0)
    center_1 = get_projected_center(group_1)

    if center_0[0] < center_1[0]:
        idxs_left = torch.where(torch.tensor(labels == 0))[0]
        idxs_right = torch.where(torch.tensor(labels == 1))[0]
    else:
        idxs_left = torch.where(torch.tensor(labels == 1))[0]
        idxs_right = torch.where(torch.tensor(labels == 0))[0]
    return idxs_left.to(cfg.device), idxs_right.to(cfg.device)

def mpc_optimize(trainer, idxs_left, idxs_right, gs_target_xyz, ctrl_target,
                 gs_init_tensor, ctrl_init_tensor, n_steps=40, n_iters=100):
    lr = 5e-2
    lambda_ctrl = 1e-3
    lambda_ctrl_target = 10.0
    best_loss = float("inf")
    best_offsets = None

    chamfer = ChamferDistance()
    ctrl_init = trainer.simulator.controller_points[0].detach().clone()

    ctrl_mask = torch.zeros(ctrl_init.shape[0], dtype=torch.bool, device=ctrl_init.device)
    ctrl_mask[idxs_left] = True
    ctrl_mask[idxs_right] = True
    num_ctrl = ctrl_mask.sum().item()

    velocities = [torch.nn.Parameter(torch.zeros(num_ctrl, 3, device=ctrl_init.device)) for _ in range(n_steps)]
    optimizer = torch.optim.Adam(velocities, lr=lr)

    for iter in tqdm(range(n_iters), desc="MPC Optimization"):
        trainer.simulator.set_init_state(
            gs_init_tensor,
            torch.zeros_like(gs_init_tensor),
            pure_inference=True
        )
        trainer.simulator.controller_points[0].data.copy_(ctrl_init_tensor)

        prev_ctrl = ctrl_init.clone()

        for t in range(n_steps):
            velocity = velocities[t]
            velocity.data = torch.clamp(velocity.data, -0.01, 0.01)  # clip to 1cm/frame

            next_ctrl = prev_ctrl.clone()
            next_ctrl[ctrl_mask] += velocity

            trainer.simulator.set_controller_interactive(prev_ctrl.detach(), next_ctrl.detach())
            trainer.simulator.step()
            
            prev_ctrl = next_ctrl

        cur_x = wp.to_torch(trainer.simulator.wp_states[-1].wp_x, requires_grad=False)
        cur_xyz = cur_x[:trainer.num_all_points].unsqueeze(0)
        target_xyz = gs_target_xyz.unsqueeze(0)
        dist1, dist2, _, _ = chamfer(cur_xyz, target_xyz)
        chamfer_loss = dist1.mean() + dist2.mean()

        final_ctrl = prev_ctrl[ctrl_mask]
        ctrl_target_loss = torch.mean((final_ctrl - ctrl_target[ctrl_mask]) ** 2)

        ctrl_reg = sum((v**2).mean() for v in velocities) / len(velocities)

        jerk_reg = sum(((velocities[t] - 2 * velocities[t - 1] + velocities[t - 2]) ** 2).mean()
                       for t in range(2, len(velocities))) / (len(velocities) - 2)
        lambda_jerk = 5.0

        loss = chamfer_loss + lambda_ctrl * ctrl_reg + lambda_ctrl_target * ctrl_target_loss + lambda_jerk * jerk_reg

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_offsets = [v.detach().clone() for v in velocities]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 20 == 0:
            print(f"Iter {iter} | Chamfer: {chamfer_loss.item():.6f} | Ctrl Reg: {ctrl_reg.item():.6f} | Ctrl Target: {ctrl_target_loss.item():.6f} | Jerk: {jerk_reg.item():.6f}")

    return best_offsets, ctrl_mask

def save_trajectory(trainer, offsets, ctrl_mask, filename="mpc_trajectory.pkl", gs_color=None):
    ctrl_init = trainer.simulator.controller_points[0].detach().clone()
    full_traj = []
    ctrl_traj = []
    prev_ctrl = ctrl_init.clone()

    for t in range(len(offsets)):
        ctrl_now = prev_ctrl.clone()
        ctrl_now[ctrl_mask] += offsets[t].detach()

        trainer.simulator.set_controller_interactive(
            trainer.simulator.controller_points[0],
            ctrl_now
        )
        if trainer.simulator.object_collision_flag:
            trainer.simulator.update_collision_graph()

        if cfg.use_graph:
            wp.capture_launch(trainer.simulator.forward_graph)
        else:
            trainer.simulator.step()

        x = wp.to_torch(trainer.simulator.wp_states[-1].wp_x, requires_grad=False).detach().cpu().numpy()
        full_traj.append(x)
        ctrl_traj.append(ctrl_now.detach().cpu().numpy())

        trainer.simulator.set_init_state(
            trainer.simulator.wp_states[-1].wp_x,
            trainer.simulator.wp_states[-1].wp_v,
            pure_inference=True
        )
        prev_ctrl = ctrl_now

    with open(filename, "wb") as f:
        pickle.dump({
            "traj": np.stack(full_traj, axis=0),
            "ctrl_traj": np.stack(ctrl_traj, axis=0),
            "gs_color": gs_color.cpu().numpy() if gs_color is not None else None,
        }, f)
    print(f"✅ Saved full particle trajectory to {filename}")

if __name__ == "__main__":
    case_name = "rope_double_hand"
    n_trajs = 10
    n_steps = 120

    trainer = load_trainer(case_name)
    ctrl_pts = trainer.simulator.controller_points[0].cpu().numpy()
    idxs_left, idxs_right = get_hand_indices(ctrl_pts, trainer.intrinsic, trainer.w2c)

    save_dir = expanduser(f"~/workspace/PhysTwin/temp_mpc_output/{case_name}")
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(os.getpid() % (2**32 - 1))
    init_dir = expanduser("~/workspace/PhysTwin/mpc_init")
    target_dir = expanduser("~/workspace/PhysTwin/mpc_target_U")

    init_plys = sorted(glob.glob(os.path.join(init_dir, "*.ply")))
    init_npys = sorted(glob.glob(os.path.join(init_dir, "*.npy")))
    target_plys = sorted(glob.glob(os.path.join(target_dir, "*.ply")))
    target_npys = sorted(glob.glob(os.path.join(target_dir, "*.npy")))

    assert len(init_plys) == len(init_npys)
    assert len(target_plys) == len(target_npys)

    # for each i, pair one random init with one target
    for i in range(n_trajs):
        init_idx = np.random.randint(len(init_plys))
        target_idx = np.random.randint(len(target_plys))
        print(f"[Pair {i}] Init: {init_idx:04d}, Target: U{target_idx:04d}")

        gs_init, _ = load_gs_xyz(init_plys[init_idx])
        ctrl_init = load_ctrl_npy(init_npys[init_idx])
        trainer.set_init_state_from_numpy(gs_init, ctrl_init)

        gs_target_path = target_plys[target_idx]
        ctrl_target_path = target_npys[target_idx]

        gs_target_xyz, gs_target_color = load_gs_xyz(gs_target_path)
        ctrl_target = load_ctrl_npy(ctrl_target_path)
        
        best_offsets, ctrl_mask = mpc_optimize(
            trainer, idxs_left, idxs_right,
            gs_target_xyz, ctrl_target,
            gs_init, ctrl_init,
            n_steps=n_steps
        )

        save_path = os.path.join(save_dir, f"mpc_traj_init{init_idx:04d}_to_U{target_idx:04d}.pkl")
        save_trajectory(trainer, best_offsets, ctrl_mask, save_path, gs_color=gs_target_color)

