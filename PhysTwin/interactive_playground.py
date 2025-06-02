# workspace/PhysTwin/interactive_playground.py
'''
python interactive_playground.py --n_ctrl_parts 2 --case_name rope_double_hand
python interactive_playground.py --n_ctrl_parts 2 --case_name rope_double_hand --replay_command_pkl temp_keyboard_commands/commands0000.pkl
python interactive_playground.py --n_ctrl_parts 2 --case_name double_lift_cloth_1 --replay_command_pkl mpc_replay/concat_commands0000.pkl
'''
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import open3d as o3d
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json
from tqdm import tqdm

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    parser.add_argument("--inv_ctrl", action="store_true")
    parser.add_argument("--replay_command_pkl", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    start_idx = args.start_idx

    # ==== 1. 加载 config ====
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    # ==== 2. 读取最佳物理参数 ====
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(optimal_path), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # ==== 3. 打印物理参数 ====
    # print("Spring stiffness:", cfg.init_spring_Y)
    # print("Dashpot damping:", cfg.dashpot_damping)
    # print("Drag damping:", cfg.drag_damping)
    # print("Collision dist:", cfg.collision_dist)
    # print("Controller radius:", cfg.controller_radius)
    # print("dt:", cfg.dt)
    # print("num_substeps:", cfg.num_substeps)

    # ==== 4. 相机参数 ====
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path

    # ==== 5. Gaussian Splatting 路径 ====
    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    # ==== 6. 初始化Trainer ====
    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]

    # ==== 7. 根据是否replay，走不同逻辑 ====
    if args.replay_command_pkl is not None:
        # ------ Replay模式 ------
        with open(args.replay_command_pkl, "rb") as f:
            replay_info = pickle.load(f)
        replay_keys = replay_info["commands"]
        start_idx = replay_info["start_idx"]

        # 组装成列表传进去
        replay_keys = [replay_keys]

        trainer.interactive_playground(
            model_path=best_model_path,
            gs_path=gaussians_path,
            n_ctrl_parts=args.n_ctrl_parts,
            inv_ctrl=args.inv_ctrl,
            replay_keys=replay_keys,
            start_idx=start_idx,
        )
    else:
        # ------ 普通键盘控制模式 ------
        trainer.interactive_playground(
            best_model_path,
            gaussians_path,
            args.n_ctrl_parts,
            args.inv_ctrl,
        )
