import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 单手动作映射
ACTION_TO_KEYS_HAND1 = {
    (1, 0, 0): ['d'], (0, 1, 0): ['w'], (-1, 0, 0): ['a'], (0, -1, 0): ['s'],
    (0, 0, 1): ['e'], (0, 0, -1): ['q']
}
ACTION_TO_KEYS_HAND2 = {
    (1, 0, 0): ['l'], (0, 1, 0): ['i'], (-1, 0, 0): ['j'], (0, -1, 0): ['k'],
    (0, 0, 1): ['o'], (0, 0, -1): ['u']
}

def get_action(delta, threshold=0.0001):
    """将连续的delta离散成(-1, 0, 1)"""
    move = []
    for d in delta:
        if d > threshold:
            move.append(1)
        elif d < -threshold:
            move.append(-1)
        else:
            move.append(0)
    return tuple(move)  # (dx, dy, dz)

import random

def generate_commands_one(init_ctrl, target_ctrl, max_steps=300):
    cmds = []

    # 拆左右手
    ctrl1 = init_ctrl[:15].copy()
    ctrl2 = init_ctrl[15:].copy()
    ctrl1_target = target_ctrl[:15]
    ctrl2_target = target_ctrl[15:]

    per_step_move = 0.003
    threshold_stop = 0.005  # 离target 5mm内就停

    for _ in range(max_steps):
        keys_this_frame = []

        delta1 = ctrl1_target.mean(axis=0) - ctrl1.mean(axis=0)
        delta2 = ctrl2_target.mean(axis=0) - ctrl2.mean(axis=0)

        if np.linalg.norm(delta1) < threshold_stop and np.linalg.norm(delta2) < threshold_stop:
            break

        def decide_keys(delta, action_keys):
            keys = []
            # x
            if delta[0] > per_step_move:
                keys.append(action_keys[(1, 0, 0)][0])
            elif delta[0] < -per_step_move:
                keys.append(action_keys[(-1, 0, 0)][0])
            # y
            if delta[1] > per_step_move:
                keys.append(action_keys[(0, 1, 0)][0])
            elif delta[1] < -per_step_move:
                keys.append(action_keys[(0, -1, 0)][0])
            # z
            if delta[2] > per_step_move:
                keys.append(action_keys[(0, 0, 1)][0])
            elif delta[2] < -per_step_move:
                keys.append(action_keys[(0, 0, -1)][0])
            return keys

        keys1 = decide_keys(delta1, ACTION_TO_KEYS_HAND1)
        keys2 = decide_keys(delta2, ACTION_TO_KEYS_HAND2)

        # 更新位置
        for key in keys1:
            if key == 'd':
                ctrl1 += np.array([per_step_move, 0, 0])
            elif key == 'a':
                ctrl1 += np.array([-per_step_move, 0, 0])
            elif key == 'w':
                ctrl1 += np.array([0, per_step_move, 0])
            elif key == 's':
                ctrl1 += np.array([0, -per_step_move, 0])
            elif key == 'e':
                ctrl1 += np.array([0, 0, per_step_move])
            elif key == 'q':
                ctrl1 += np.array([0, 0, -per_step_move])

        for key in keys2:
            if key == 'l':
                ctrl2 += np.array([per_step_move, 0, 0])
            elif key == 'j':
                ctrl2 += np.array([-per_step_move, 0, 0])
            elif key == 'i':
                ctrl2 += np.array([0, per_step_move, 0])
            elif key == 'k':
                ctrl2 += np.array([0, -per_step_move, 0])
            elif key == 'o':
                ctrl2 += np.array([0, 0, per_step_move])
            elif key == 'u':
                ctrl2 += np.array([0, 0, -per_step_move])

        keys_this_frame = keys1 + keys2
        cmds.append(keys_this_frame)

    return cmds



def generate_keyboard_commands(save_dir, n_trajs=100):
    """
    主函数：随机采样 + 生成commands
    """
    init_dir = Path("~/workspace/PhysTwin/mpc_init").expanduser()
    target_dir = Path("~/workspace/PhysTwin/mpc_target_U").expanduser()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    init_files = sorted(init_dir.glob("*.npy"))
    target_files = sorted(target_dir.glob("*.npy"))

    for traj_idx in tqdm(range(n_trajs)):
        init_path = np.random.choice(init_files)
        target_path = np.random.choice(target_files)

        init_ctrl = np.load(init_path)
        target_ctrl = np.load(target_path)

        commands = generate_commands_one(init_ctrl, target_ctrl)

        save_path = save_dir / f"commands{traj_idx:04d}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(commands, f)

        print(f"✅ Saved {save_path.name}")

if __name__ == "__main__":
    save_dir = "~/workspace/PhysTwin/temp_keyboard_commands"
    generate_keyboard_commands(os.path.expanduser(save_dir), n_trajs=20)
