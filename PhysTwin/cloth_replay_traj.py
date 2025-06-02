# cloth_replay_traj.py
import numpy as np
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import argparse
import pickle
import numpy as np
from pathlib import Path

"""
python cloth_replay_traj.py --command_pkl ~/workspace/PhysTwin/temp_keyboard_commands_cloth/commands0000.pkl
"""

def concat_fixed_traj(replay_pkl_path, fixed_pkl_path):
    # 加载 replay轨迹
    with open(replay_pkl_path, 'rb') as f:
        replay_data = pickle.load(f)
    replay_ctrl_traj = replay_data['ctrl_traj']
    replay_gs_traj = replay_data['traj']

    # 加载固定轨迹
    with open(fixed_pkl_path, 'rb') as f:
        fixed_data = pickle.load(f)
    fixed_ctrl_traj = fixed_data['ctrl_traj']
    fixed_gs_traj = fixed_data['traj']

    # === 对齐 fixed_traj 的第一帧到 replay_traj 的最后一帧 ===
    last_gs_replay = replay_gs_traj[-1]    # (N, 3)
    first_gs_fixed = fixed_gs_traj[0]       # (N, 3)

    center_replay = last_gs_replay.mean(axis=0)
    center_fixed = first_gs_fixed.mean(axis=0)
    offset = center_replay - center_fixed

    print(f"🧩 Aligning fixed traj by offset: {offset}")

    # 应用平移
    fixed_gs_traj += offset[None, None, :]  # (T, N, 3)
    fixed_ctrl_traj += offset[None, None, :]

    # 拼接
    new_ctrl_traj = np.concatenate([replay_ctrl_traj, fixed_ctrl_traj], axis=0)
    new_gs_traj = np.concatenate([replay_gs_traj, fixed_gs_traj], axis=0)

    # 保存
    save_path = replay_pkl_path.parent / ("concat_" + replay_pkl_path.name)
    with open(save_path, 'wb') as f:
        pickle.dump({
            'ctrl_traj': new_ctrl_traj,
            'traj': new_gs_traj,
        }, f)
    print(f"✅ Saved concatenated traj to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command_pkl", type=str, required=True)
    args = parser.parse_args()

    # 🆕 从输入的 command_pkl 文件名中解析出数字
    command_name = os.path.basename(args.command_pkl)  # e.g., commands0023.pkl
    idx = int(command_name.replace("commands", "").replace(".pkl", ""))  # 23

    # 1. 先跑replay
    cmd = f"python interactive_playground.py --n_ctrl_parts 2 --case_name double_lift_cloth_1 --replay_command_pkl {args.command_pkl} --start_idx {idx}"
    os.system(cmd)

    # 2. 找到replay之后生成的pkl
    replay_dir = Path("mpc_replay")
    replay_pkl = replay_dir / f"commands{idx:04d}.pkl"
    if not replay_pkl.exists():
        print(f"❌ Replay traj {replay_pkl} not found, skip.")
        exit(0)
    print(f"📦 Found replay pkl: {replay_pkl}")

    # 3. 加载固定traj
    fixed_pkl = Path("fixed_traj/fixed_traj.pkl")  # 这里是你固定轨迹的路径

    # 4. 拼接
    concat_fixed_traj(replay_pkl, fixed_pkl)
