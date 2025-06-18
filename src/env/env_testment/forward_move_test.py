# /workspace/src/env/env_testment/forward_move_test.py
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PhysTwin")))
from src.env.phystwin_env import PhysTwinEnv

def main():
    env = PhysTwinEnv()
    env.reset_to_origin(n_ctrl_parts=2)

    num_steps = 20
    delta = np.array([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])  # 每个手沿X轴移动

    gs_traj = []
    ctrl_traj = []

    for step in range(num_steps):
        env.step(n_ctrl_parts=2, action=delta)
        obs = env.get_obs()
        gs_traj.append(obs["state"])
        ctrl_traj.append(obs["ctrl_pts"])

    gs_traj = np.array(gs_traj)         # [T, N_pts, 3]
    ctrl_traj = np.array(ctrl_traj)     # [T, N_ctrl, 3]

    # ✅ 可视化最后一帧（GS + 控制点） 
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("Final Frame: GS + Control Points")
    ax.scatter(gs_traj[-1][:, 0], gs_traj[-1][:, 1], gs_traj[-1][:, 2], s=1, label="GS Points")
    ax.scatter(ctrl_traj[-1][:, 0], ctrl_traj[-1][:, 1], ctrl_traj[-1][:, 2], color="red", label="Ctrl Points")
    ax.legend()
    plt.show()

    # ✅ 可视化控制点轨迹
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.set_title("Control Point Trajectories")
    for i in range(ctrl_traj.shape[1]):
        traj = ctrl_traj[:, i, :]
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Ctrl Point {i}")
    ax2.legend()
    plt.show()

if __name__ == "__main__":
    main()
