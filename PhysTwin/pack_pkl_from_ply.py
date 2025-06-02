import os
import argparse
import pickle
import numpy as np
from plyfile import PlyData
from pathlib import Path

"""
python pack_pkl_from_ply.py \
  --init_dir ~/workspace/PhysTwin/mpc_init \
  --out ~/workspace/PhysTwin/mpc_replay/concat_commands0000.pkl \
  --start 0 \
  --end 39
"""

def load_gs_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    points = np.stack([plydata['vertex'][axis] for axis in ['x', 'y', 'z']], axis=1)
    return points.astype(np.float32)

def pack_traj(init_dir, start_idx, end_idx, out_path):
    traj = []
    ctrl_traj = []

    for i in range(start_idx, end_idx + 1):
        ply_file = os.path.join(init_dir, f"{i:04d}.ply")
        npy_file = os.path.join(init_dir, f"{i:04d}.npy")

        if not os.path.exists(ply_file) or not os.path.exists(npy_file):
            print(f"⚠️ Missing frame {i:04d}. Skipping.")
            continue

        gs = load_gs_from_ply(ply_file)
        ctrl = np.load(npy_file).astype(np.float32)

        traj.append(gs)
        ctrl_traj.append(ctrl)

    if not traj or not ctrl_traj:
        raise RuntimeError("❌ No valid frames found.")

    traj = np.stack(traj, axis=0)          # (T, N, 3)
    ctrl_traj = np.stack(ctrl_traj, axis=0)  # (T, M, 3)

    output = {
        "traj": traj,
        "ctrl_traj": ctrl_traj
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"✅ Saved {traj.shape[0]} frames to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_dir", type=str, required=True, help="Folder containing mpc_init/*.ply and *.npy")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output pkl path")
    args = parser.parse_args()

    pack_traj(args.init_dir, args.start, args.end, args.out)
