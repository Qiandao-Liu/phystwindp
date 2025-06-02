# rope_replay_traj.py
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command_pkl", type=str, required=True)
    args = parser.parse_args()

    cmd = f"python interactive_playground.py --n_ctrl_parts 2 --case_name rope_double_hand --replay_command_pkl {args.command_pkl}"
    os.system(cmd)

