# batch_replay.py
import os
import subprocess
from pathlib import Path

commands_dir = Path("~/workspace/PhysTwin/temp_keyboard_commands_cloth").expanduser()

for idx in range(200):
    pkl_name = f"commands{idx:04d}.pkl"
    pkl_path = commands_dir / pkl_name

    if not pkl_path.exists():
        print(f"❌ {pkl_path} 不存在，跳过")
        continue

    cmd = f"python cloth_replay_traj.py --command_pkl {pkl_path}"
    print(f"🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True)
