# workspace/src/planning/gradient_mpc_new.py
import torch
import warp as wp
from tqdm import trange
import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)

def main(case_name="double_lift_cloth_1", init_idx=0, target_idx=0):
    init_path = f"PhysTwin/mpc_init/init_{init_idx:03d}.pkl"
    target_path = f"PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"

    with open(init_path, "rb") as f:
        init_data = pickle.load(f)
    with open(target_path, "rb") as f:
        target_data = pickle.load(f)
