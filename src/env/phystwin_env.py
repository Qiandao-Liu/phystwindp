# /workspace/src/env/phystwin_env.py

import numpy as np
import torch
from PhysTwin.qqtt.engine.trainer_warp import InvPhyTrainerWarp
from PhysTwin.qqtt.utils import cfg
import pickle, json, os

class PhysTwinEnv:
    def __init__(self, case_name="double_lift_cloth_1", base_path="./data/different_types", gaussian_path="./gaussian_output", pure_inference=True):
        self.case_name = case_name
        self.base_path = base_path
        self.gaussian_path = gaussian_path
        self.pure_inference = pure_inference

        # ======== Load config based on task =========
        if "cloth" in case_name or "package" in case_name:
            cfg.load_from_yaml("configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/real.yaml")

        self._load_camera_calibration()
        self._load_optimal_params()

        # ======== Construct simulator =========
        self.trainer = InvPhyTrainerWarp(
            data_path=f"{self.base_path}/{case_name}/final_data.pkl",
            base_dir=f"./temp_experiments/{case_name}",
            pure_inference_mode=pure_inference
        )

        self.sim = self.trainer.simulator
        self.ctrl_pts = None
        self.gs_pts = None
        self.step_id = 0

    def _load_camera_calibration(self):
        # Calibrate views (for rendering, optional)
        with open(f"{self.base_path}/{self.case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(f"{self.base_path}/{self.case_name}/metadata.json", "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]

    def _load_optimal_params(self):
        optimal_path = f"./experiments_optimization/{self.case_name}/optimal_params.pkl"
        assert os.path.exists(optimal_path), f"Missing: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    def reset(self, init_ctrl_pts=None, init_obj_pts=None):
        # Load from data if not provided
        if init_ctrl_pts is None or init_obj_pts is None:
            ctrl_all = self.trainer.ctrl_points_all
            obj_all = self.trainer.gs_points_all
            init_ctrl_pts = ctrl_all[0]
            init_obj_pts = obj_all[0]

        self.ctrl_pts = init_ctrl_pts
        self.gs_pts = init_obj_pts
        self.sim.set_init_state_from_numpy(self.ctrl_pts, self.gs_pts)
        self.step_id = 0

        return self.get_obs()

    def step(self, delta_ctrl):
        # Apply delta and step physics
        self.step_id += 1
        self.sim.step_ctrl(delta_ctrl)
        self.ctrl_pts = self.sim.get_ctrl_pts()
        self.gs_pts = self.sim.get_obj_pts()
        return self.get_obs()

    def get_obs(self):
        return {
            "ctrl_pts": self.ctrl_pts.copy(),
            "gs_pts": self.gs_pts.copy(),
        }

    def render(self):
        # You can optionally add a pointcloud visualizer here
        print(f"[Render] Step {self.step_id} | ctrl shape: {self.ctrl_pts.shape} | obj shape: {self.gs_pts.shape}")

    def get_state(self):
        return self.ctrl_pts.copy(), self.gs_pts.copy()

    def set_state(self, ctrl_pts, obj_pts):
        self.ctrl_pts = ctrl_pts
        self.gs_pts = obj_pts
        self.sim.set_init_state_from_numpy(ctrl_pts, obj_pts)
