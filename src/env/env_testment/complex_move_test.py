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
    ctrl_delta_traj = np.zeros((T, 2, 3))

    # 控制点0：圆周运动（XZ）
    ctrl_delta_traj[:, 0, 0] = 0.01 * np.cos(theta)
    ctrl_delta_traj[:, 0, 2] = 0.01 * np.sin(theta)

    # 控制点1：Y方向来回 + X方向轻微波动
    ctrl_delta_traj[:, 1, 1] = 0.01 * np.sin(2 * theta)
    ctrl_delta_traj[:, 1, 0] = 0.005 * np.sin(3 * theta)

    gs_traj = []
    ctrl_traj = []

    for t in range(T):
        env.step(n_ctrl_parts=2, action=ctrl_delta_traj[t])
        obs = env.get_obs()
        gs_traj.append(obs["state"])
        ctrl_traj.append(obs["ctrl_pts"])

    animate_gs_trajectory_v(np.array(gs_traj), np.array(ctrl_traj), save_path="complex_traj.mp4")

if __name__ == "__main__":
    main()
