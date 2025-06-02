# cloth_generate_traj.py
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

ACTION_TO_KEYS_HAND1 = {
    (1, 0, 0): ['d'], (0, 1, 0): ['w'], (-1, 0, 0): ['a'], (0, -1, 0): ['s'],
    (0, 0, 1): ['e'], (0, 0, -1): ['q']
}
ACTION_TO_KEYS_HAND2 = {
    (1, 0, 0): ['l'], (0, 1, 0): ['i'], (-1, 0, 0): ['j'], (0, -1, 0): ['k'],
    (0, 0, 1): ['o'], (0, 0, -1): ['u']
}

def generate_commands_one(default_ctrl, init_ctrl, target_ctrl, max_steps_per_stage=300):
    """
    生成两段命令:
    - 段1: default_ctrl -> init_ctrl
    - 段2: init_ctrl -> target_ctrl
    返回: 全部commands list of list & 起始idx int
    """
    cmds = []

    ctrl1 = default_ctrl[:15].copy()
    ctrl2 = default_ctrl[15:].copy()
    ctrl1_init = init_ctrl[:15]
    ctrl2_init = init_ctrl[15:]
    ctrl1_target = target_ctrl[:15]
    ctrl2_target = target_ctrl[15:]

    per_step_move = 0.005
    threshold_stop = 0.02

    def move_one_stage(ctrl1, ctrl2, ctrl1_goal, ctrl2_goal, max_steps):
        stage_cmds = []
        step_idx = 0
        for _ in range(max_steps):
            keys_this_frame = []

            delta1 = (ctrl1_goal.mean(axis=0) - ctrl1.mean(axis=0))
            delta2 = (ctrl2_goal.mean(axis=0) - ctrl2.mean(axis=0))
            print(f"[Gen Traj] Step {step_idx}: delta1 norm {np.linalg.norm(delta1):.5f}, delta2 norm {np.linalg.norm(delta2):.5f}")

            delta1_norm = np.linalg.norm(delta1)
            delta2_norm = np.linalg.norm(delta2)
            print(f"[Gen Traj] Step {step_idx}: delta1 norm {delta1_norm:.5f}, delta2 norm {delta2_norm:.5f}")

            # early stop
            if delta1_norm < threshold_stop and delta2_norm < threshold_stop:
                print(f"[Gen Traj] 🎯 Early stop at step {step_idx}: delta1 {delta1_norm:.5f}, delta2 {delta2_norm:.5f}")
                break

            def decide_keys(delta, action_keys):
                keys = []
                if delta[0] > per_step_move:
                    keys.append(action_keys[(1, 0, 0)][0])
                elif delta[0] < -per_step_move:
                    keys.append(action_keys[(-1, 0, 0)][0])
                if delta[1] > per_step_move:
                    keys.append(action_keys[(0, 1, 0)][0])
                elif delta[1] < -per_step_move:
                    keys.append(action_keys[(0, -1, 0)][0])
                if delta[2] > per_step_move:
                    keys.append(action_keys[(0, 0, 1)][0])
                elif delta[2] < -per_step_move:
                    keys.append(action_keys[(0, 0, -1)][0])
                return keys

            keys1 = decide_keys(delta1, ACTION_TO_KEYS_HAND1)
            keys2 = decide_keys(delta2, ACTION_TO_KEYS_HAND2)

            # 更新位置
            for key in keys1:
                if key == 'd': ctrl1 += np.array([per_step_move, 0, 0])
                if key == 'a': ctrl1 += np.array([-per_step_move, 0, 0])
                if key == 'w': ctrl1 += np.array([0, per_step_move, 0])
                if key == 's': ctrl1 += np.array([0, -per_step_move, 0])
                if key == 'e': ctrl1 += np.array([0, 0, per_step_move])
                if key == 'q': ctrl1 += np.array([0, 0, -per_step_move])

            for key in keys2:
                if key == 'l': ctrl2 += np.array([per_step_move, 0, 0])
                if key == 'j': ctrl2 += np.array([-per_step_move, 0, 0])
                if key == 'i': ctrl2 += np.array([0, per_step_move, 0])
                if key == 'k': ctrl2 += np.array([0, -per_step_move, 0])
                if key == 'o': ctrl2 += np.array([0, 0, per_step_move])
                if key == 'u': ctrl2 += np.array([0, 0, -per_step_move])

            stage_cmds.append(keys1 + keys2)
            step_idx += 1

        return stage_cmds, ctrl1, ctrl2

    # 第一段: default ➔ init
    stage1_cmds, ctrl1, ctrl2 = move_one_stage(ctrl1, ctrl2, ctrl1_init, ctrl2_init, max_steps_per_stage)
    
    # 第二段: init ➔ target
    stage2_cmds, _, _ = move_one_stage(ctrl1, ctrl2, ctrl1_target, ctrl2_target, max_steps_per_stage)

    full_cmds = stage1_cmds + stage2_cmds
    start_idx_of_stage2 = len(stage1_cmds)

    return full_cmds, start_idx_of_stage2


def generate_keyboard_commands(save_dir, n_trajs=100):
    init_dir = Path("~/workspace/PhysTwin/mpc_init_cloth").expanduser()
    target_dir = Path("~/workspace/PhysTwin/mpc_target_fold").expanduser()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    init_files = sorted(init_dir.glob("*.npy"))
    target_files = sorted(target_dir.glob("*.npy"))

    default_ctrl_path = Path("~/workspace/PhysTwin/mpc_default_state/default_cloth_1_ctrl.npy").expanduser()
    default_ctrl = np.load(default_ctrl_path)
    print(default_ctrl.shape)  # 应该是 (30, 3)
    print(default_ctrl[:5])    # 打印一下前几行看看是不是手拿着方巾两个角

    for traj_idx in tqdm(range(n_trajs)):
        init_path = np.random.choice(init_files)
        target_path = np.random.choice(target_files)

        init_ctrl = np.load(init_path)
        target_ctrl = np.load(target_path)

        commands, start_idx_of_stage2 = generate_commands_one(default_ctrl, init_ctrl, target_ctrl)

        save_path = save_dir / f"commands{traj_idx:04d}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({
                "commands": commands,
                "start_idx": start_idx_of_stage2
            }, f)

        print(f"✅ Saved {save_path.name}")


if __name__ == "__main__":
    save_dir = "~/workspace/PhysTwin/temp_keyboard_commands_cloth"
    generate_keyboard_commands(os.path.expanduser(save_dir), n_trajs=100)
