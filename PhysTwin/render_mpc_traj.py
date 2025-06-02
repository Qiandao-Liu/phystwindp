import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import glob

def render_trajectory(traj_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(traj_path, "rb") as f:
        data = pickle.load(f)
    traj = data["traj"]
    ctrl_traj = data.get("ctrl_traj", None)
    T, N, _ = traj.shape

    initial = traj[0]
    final = traj[-1]
    displacement = np.linalg.norm(final - initial, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c=[], cmap='viridis', s=10, vmin=displacement.min(), vmax=displacement.max())
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

    # 平均位移图
    framewise_disp = [np.linalg.norm(traj[i] - traj[i - 1], axis=1).mean() for i in range(1, T)]
    plt.figure()
    plt.plot(range(1, T), framewise_disp, marker='o')
    plt.title("Average Particle Displacement per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Displacement (m)")
    plt.grid(True)
    chart_path = os.path.join(save_dir, "framewise_displacement.png")
    plt.savefig(chart_path, dpi=200)
    print(f"📈 Chart saved to {chart_path}")
    plt.close()

    # 距离目标图
    target_pos = traj[-1]
    dist_to_target = np.linalg.norm(traj - target_pos[None, :, :], axis=2).mean(axis=1)
    plt.figure()
    plt.plot(range(T), dist_to_target, marker='o')
    plt.title("Mean Distance to Target per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Distance (m)")
    chart_path = os.path.join(save_dir, "framewise_target_distance.png")
    plt.savefig(chart_path, dpi=200)
    print(f"📈 Target convergence chart saved to {chart_path}")
    plt.close()

# === 批量处理所有 MPC 轨迹 ===
mpc_dir = os.path.expanduser("~/workspace/PhysTwin/temp_mpc_output/rope_double_hand")

traj_files = sorted(glob.glob(os.path.join(mpc_dir, "mpc_trajectory_*.pkl")))

for traj_path in traj_files:
    traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    save_dir = os.path.expanduser(f"~/workspace/renders/{traj_name}")
    render_trajectory(traj_path, save_dir)
