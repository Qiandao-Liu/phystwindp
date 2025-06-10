# /workspace/src/env/env_testment/forward_move_test.py
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv
import numpy as np
import open3d as o3d

env = PhysTwinEnv(case_name="double_lift_cloth_1")
obs = env.reset()

print("✅ Initial obs:", {k: v.shape for k, v in obs.items()})

delta_dir = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(30, 1)  # (30, 3)
delta_mag = 0.005

delta = delta_dir * delta_mag

# 初始化 Open3D 可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="GS Movement", width=960, height=720)

# 初始化两个点云（GS点红色，控制点绿色）
gs_pcd = o3d.geometry.PointCloud()
gs_pcd.points = o3d.utility.Vector3dVector(obs["gs_pts"].cpu().numpy())
gs_pcd.paint_uniform_color([1, 0, 0])
vis.add_geometry(gs_pcd)

ctrl_pcd = o3d.geometry.PointCloud()
ctrl_pcd.points = o3d.utility.Vector3dVector(obs["ctrl_pts"].cpu().numpy())
ctrl_pcd.paint_uniform_color([0, 1, 0])
vis.add_geometry(ctrl_pcd)

# 循环执行 step + 可视化更新
for i in range(200):
    obs = env.step(delta)

    if i % 1 == 0:
        env.render()

        # 更新可视化点位置
        gs_pcd.points = o3d.utility.Vector3dVector(obs["gs_pts"].cpu().numpy())
        ctrl_pcd.points = o3d.utility.Vector3dVector(obs["ctrl_pts"].cpu().numpy())
        vis.update_geometry(gs_pcd)
        vis.update_geometry(ctrl_pcd)
        vis.poll_events()
        vis.update_renderer()

        # 打印运动信息
        if i > 0:
            diff = obs["gs_pts"] - prev_gs
            print(f"Step {i:03d} | Δgs mean: {diff.norm(dim=1).mean():.6f}")
            print(f"🔴 Step {i:03d} | First GS point: {obs['gs_pts'][0].tolist()}")
        prev_gs = obs["gs_pts"].clone()

# vis.destroy_window()