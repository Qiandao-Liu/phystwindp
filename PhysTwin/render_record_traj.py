# ~/workspace/PhysTwin/render_record_traj.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

import glob

mpc_dir = os.path.expanduser("~/workspace/PhysTwin/mpc_targets")
all_ply = sorted(glob.glob(os.path.join(mpc_dir, "*.ply")))
all_npy = sorted(glob.glob(os.path.join(mpc_dir, "*.npy")))

traj = []
ctrl_traj = []
for ply_path, npy_path in zip(all_ply, all_npy):
    # ✅ 使用 Open3D 读取 .ply 中的点
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points)
    traj.append(xyz)

    # ✅ 读取控制点
    ctrl_traj.append(np.load(npy_path))

traj = np.stack(traj)
ctrl_traj = np.stack(ctrl_traj)
T, N, _ = traj.shape

save_dir = os.path.expanduser("~/workspace/renders/mpc_targets_preview")
os.makedirs(save_dir, exist_ok=True)

# === displacement coloring ===
initial = traj[0]
final = traj[-1]
displacement = np.linalg.norm(final - initial, axis=1)

# === 渲染动画 ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c=[], cmap='viridis', s=10, vmin=displacement.min(), vmax=displacement.max())

# 初始化阶段创建 ctrl_scatter，注意 color 设为 red
ctrl_scatter = ax.scatter([], [], [], color='red', s=30)

def init():
    norm = plt.Normalize(vmin=displacement.min(), vmax=displacement.max())
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.5, label="Total Displacement (m)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1.0)
    ax.set_title('MPC Trajectory')
    ax.scatter([0], [0], [0], color='black', s=100, label='origin')
    return sc, ctrl_scatter

def update(i):
    pos = traj[i]
    sc._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    sc.set_array(displacement)

    if ctrl_traj is not None:
        ctrl_pos = ctrl_traj[i]
        ctrl_scatter._offsets3d = (ctrl_pos[:, 0], ctrl_pos[:, 1], ctrl_pos[:, 2])

    ax.set_title(f"Frame {i+1}/{T}")
    return sc, ctrl_scatter

ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=100, blit=False)
out_path = os.path.join(save_dir, 'mpc_traj.mp4')
ani.save(out_path, fps=20, dpi=200)
print(f"🎞️ Saved video to {out_path}")
plt.close()
