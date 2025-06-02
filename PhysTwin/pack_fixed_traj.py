import os
import pickle
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

def load_gs_points(ply_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return np.asarray(pcd.points, dtype=np.float32)

def load_ctrl_points(npy_path):
    return np.load(npy_path).astype(np.float32)

def pack_fixed_traj(input_dir, output_pkl):
    input_dir = Path(input_dir)
    gs_frames = []
    ctrl_frames = []

    ply_files = sorted(input_dir.glob("*.ply"))
    npy_files = sorted(input_dir.glob("*.npy"))

    assert len(ply_files) == len(npy_files), "Mismatch between ply and npy frames!"

    for ply_file, npy_file in tqdm(zip(ply_files, npy_files), total=len(ply_files)):
        gs_pts = load_gs_points(ply_file)
        ctrl_pts = load_ctrl_points(npy_file)

        gs_frames.append(gs_pts)
        ctrl_frames.append(ctrl_pts)

    traj = np.stack(gs_frames, axis=0)        # (N_frames, N_points, 3)
    ctrl_traj = np.stack(ctrl_frames, axis=0)  # (N_frames, N_ctrl_points, 3)

    with open(output_pkl, 'wb') as f:
        pickle.dump({'traj': traj, 'ctrl_traj': ctrl_traj}, f)

    print(f"✅ Packed fixed traj to {output_pkl}")
    print(f"Total frames: {len(gs_frames)}, GS points per frame: {traj.shape[1]}, Ctrl points per frame: {ctrl_traj.shape[1]}")

if __name__ == "__main__":
    pack_fixed_traj(
        input_dir="mpc_init",                       # 你固定traj的ply+npy目录
        output_pkl="fixed_traj/fixed_traj.pkl"       # 输出保存成一个pkl
    )
