# /workspace/src/env/env_testment/complex_move_test.py
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PhysTwin")))
from src.env.phystwin_env import PhysTwinEnv

def animate_gs_trajectory_v(gs_traj, ctrl_traj, save_path=None):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    scat_gs = ax.scatter([], [], [], s=1, color="blue", label="GS")
    scat_ctrl = ax.scatter([], [], [], s=20, color="red", label="Ctrl")

    def init():
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([0.3, -0.1])
        return scat_gs, scat_ctrl

    def update(frame):
        gs = gs_traj[frame]
        ctrl = ctrl_traj[frame]
        scat_gs._offsets3d = (gs[:, 0], gs[:, 1], gs[:, 2])
        scat_ctrl._offsets3d = (ctrl[:, 0], ctrl[:, 1], ctrl[:, 2])
        ax.set_title(f"Frame {frame}")
        return scat_gs, scat_ctrl

    ani = animation.FuncAnimation(fig, update, frames=len(gs_traj), init_func=init, interval=50)
    plt.legend()

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=30)
        print(f"🎞️ 动画已保存为: {save_path}")
    else:
        plt.show()

def main():
    env = PhysTwinEnv()
    env.reset_to_origin(n_ctrl_parts=2)

    T = 200
    theta = np.linspace(0, 2 * np.pi, T)
    ctrl_delta_traj_1 = np.zeros((T, 2, 3))
    ctrl_delta_traj_1[:, 0, 0] = 0.01 * np.cos(theta)
    ctrl_delta_traj_1[:, 0, 2] = 0.01 * np.sin(theta)
    ctrl_delta_traj_1[:, 1, 1] = 0.01 * np.sin(2 * theta)
    ctrl_delta_traj_1[:, 1, 0] = 0.005 * np.sin(3 * theta)

    gs_traj = []
    ctrl_traj = []

    # 第一段控制
    for t in range(T):
        env.step(n_ctrl_parts=2, action=ctrl_delta_traj_1[t])
        obs = env.get_obs()
        gs_traj.append(obs["state"])
        ctrl_traj.append(obs["ctrl_pts"])

    # ✅ Reset：重置模拟器状态
    env.reset_to_origin(n_ctrl_parts=2)

    # 第二段控制：不同方向的拉扯运动
    ctrl_delta_traj_2 = np.zeros((T, 2, 3))
    ctrl_delta_traj_2[:, 0, 1] = 0.01 * np.sin(theta)  # 控制点0在Y方向拉扯
    ctrl_delta_traj_2[:, 1, 2] = -0.01 * np.cos(theta) # 控制点1在Z方向上挤压

    for t in range(T):
        env.step(n_ctrl_parts=2, action=ctrl_delta_traj_2[t])
        obs = env.get_obs()
        gs_traj.append(obs["state"])
        ctrl_traj.append(obs["ctrl_pts"])

    # 可视化合并后的整个轨迹（共2T帧）
    animate_gs_trajectory_v(np.array(gs_traj), np.array(ctrl_traj), save_path="complex_traj_reset_twice.mp4")


if __name__ == "__main__":
    main()
