import os
import pickle
import numpy as np
import open3d as o3d

def save_first_frame(pkl_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 加载
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traj = data.get("gs_traj", data.get("traj"))
    ctrl_traj = data["ctrl_traj"]

    # 取第0帧
    first_gs = traj[0]      # (N_points, 3)
    first_ctrl = ctrl_traj[0]  # (N_ctrl, 3)

    # 保存GS为ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_gs)
    ply_path = os.path.join(save_dir, "target_frame.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✅ Saved GS points to {ply_path}")

    # 保存ctrl为npy
    npy_path = os.path.join(save_dir, "target_ctrl.npy")
    np.save(npy_path, first_ctrl)
    print(f"✅ Saved control points to {npy_path}")

if __name__ == "__main__":
    save_first_frame(
        pkl_path="fixed_traj/fixed_traj.pkl",   # 你的固定traj
        save_dir="mpc_target_fold"              # 生成的目标目录
    )
