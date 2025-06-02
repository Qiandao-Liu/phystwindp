import os
import sys
import torch
import numpy as np
from run_mpc_rope import (
    load_trainer, mpc_optimize, get_hand_indices,
    load_gs_xyz, load_ctrl_npy, save_trajectory
)
from qqtt.utils import cfg

def main(task_indices):
    case_name = "rope_double_hand"

    save_dir = os.path.expanduser(f"~/workspace/PhysTwin/temp_mpc_output/{case_name}")
    os.makedirs(save_dir, exist_ok=True)

    # 目标数据
    target_dir = os.path.expanduser("~/workspace/PhysTwin/mpc_target_U")
    target_plys = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".ply")])
    target_npys = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".npy")])
    assert len(target_plys) == len(target_npys)

    # 加载手部索引
    trainer = load_trainer(case_name)
    ctrl_pts = trainer.simulator.controller_points[0].cpu().numpy()
    idxs_left, idxs_right = get_hand_indices(ctrl_pts, trainer.intrinsic, trainer.w2c)

    for i in task_indices:
        init_idx = np.random.randint(len(init_plys))
        target_idx = np.random.randint(len(target_plys))
        print(f"[Pair {i}] Init: {init_idx:04d}, Target: U{target_idx:04d}")

        gs_init_tensor, _ = load_gs_xyz(init_plys[init_idx])
        ctrl_init_tensor = load_ctrl_npy(init_npys[init_idx])

        gs_target_xyz, gs_color = load_gs_xyz(target_plys[target_idx])
        ctrl_target = load_ctrl_npy(target_npys[target_idx])

        print(f"[{i}] Init stats: y range = {gs_init_tensor[:,1].min().item():.3f} ~ {gs_init_tensor[:,1].max().item():.3f}")
        print(f"[{i}] Target stats: y range = {gs_target_xyz[:,1].min().item():.3f} ~ {gs_target_xyz[:,1].max().item():.3f}")

        offsets, ctrl_mask = mpc_optimize(
            trainer, idxs_left, idxs_right,
            gs_target_xyz.to(cfg.device),
            ctrl_target.to(cfg.device),
            gs_init_tensor.to(cfg.device),
            ctrl_init_tensor.to(cfg.device),
            n_steps=120, n_iters=100
        )
        save_path = os.path.join(save_dir, f"mpc_trajectory_{i:04d}.pkl")
        save_trajectory(trainer, offsets, ctrl_mask, save_path, gs_color=gs_color.to(cfg.device))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    task_indices = list(range(args.start, args.end + 1))

    init_dir = os.path.expanduser("~/workspace/PhysTwin/mpc_init")
    init_plys = sorted([os.path.join(init_dir, f) for f in os.listdir(init_dir) if f.endswith(".ply")])
    init_npys = sorted([os.path.join(init_dir, f) for f in os.listdir(init_dir) if f.endswith(".npy")])
    assert len(init_plys) == len(init_npys)

    main(task_indices)
