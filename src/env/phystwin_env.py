# /workspace/src/env/phystwin_env.py
import os
import pickle
import json
import glob
import torch
import numpy as np

from PhysTwin.qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp
from PhysTwin.qqtt.engine.trainer_warp import InvPhyTrainerWarp  # only used for _init_start()
from PhysTwin.qqtt.utils import cfg

from importlib import reload
from PhysTwin.qqtt import utils
cfg = reload(utils).cfg

print("✅ [DEBUG] cfg BEFORE load:", vars(cfg))  # 应该是空



class PhysTwinEnv:
    def __init__(
        self,
        case_name="double_lift_cloth_1",
        base_path=None,
        gaussian_path=None,
        pure_inference=False,
    ):
        self.case_name = case_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pure_inference = pure_inference

        # === Paths ===
        self.base_path = base_path or os.path.join(
            os.path.dirname(__file__), "../../PhysTwin/data/different_types"
        )
        self.gaussian_path = gaussian_path or os.path.join(
            os.path.dirname(__file__), "../../PhysTwin/gaussian_output"
        )
        self.data_path = os.path.join(self.base_path, case_name, "final_data.pkl")
        self.exp_base_dir = f"./temp_experiments/{case_name}"
        self.optimal_param_path = os.path.join(
            os.path.dirname(__file__),
            f"../../PhysTwin/experiments_optimization/{case_name}/optimal_params.pkl"
        )
        self.model_ckpt_glob = os.path.join(
            os.path.dirname(__file__),
            f"../../PhysTwin/experiments/{case_name}/train/best_*.pth"
        )

        # === Load config FIRST ===
        config_dir = os.path.join(os.path.dirname(__file__), "../../PhysTwin/configs")
        if "cloth" in case_name or "package" in case_name:
            cfg.load_from_yaml(os.path.join(config_dir, "cloth.yaml"))
        else:
            cfg.load_from_yaml(os.path.join(config_dir, "real.yaml"))
        print("✅ [DEBUG] cfg AFTER load:", vars(cfg))  # 应该包含 init_spring_Y 等字段


        # ✅ THEN set these after load_from_yaml()
        self.data_path = os.path.join(self.base_path, case_name, "final_data.pkl")
        self.exp_base_dir = f"./temp_experiments/{case_name}"

        cfg.data_path = self.data_path
        print("✅ [DEBUG] cfg AFTER patch:", cfg.data_path)
        cfg.base_dir = self.exp_base_dir
        cfg.device = self.device
        cfg.run_name = case_name

        # ✅ Set critical config fields early
        cfg.data_path = self.data_path
        cfg.base_dir = self.exp_base_dir
        cfg.device = self.device
        cfg.run_name = case_name

        # === Load auxiliary components ===
        self._load_camera_calibration()
        self._load_optimal_params()
        self._load_and_build_simulator()
        self._load_trained_model()

        # === Runtime states ===
        self.ctrl_pts = None
        self.gs_pts = None
        self.step_id = 0

    def _load_camera_calibration(self):
        calib_path = os.path.join(self.base_path, self.case_name, "calibrate.pkl")
        meta_path = os.path.join(self.base_path, self.case_name, "metadata.json")
        if not os.path.exists(calib_path) or not os.path.exists(meta_path):
            return
        with open(calib_path, "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(meta_path, "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]
        cfg.bg_img_path = data.get("bg_img_path", "")

    def _load_optimal_params(self):
        assert os.path.exists(self.optimal_param_path), f"Missing: {self.optimal_param_path}"
        with open(self.optimal_param_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    def _load_and_build_simulator(self):
        # Load and expose dataset
        with open(self.data_path, "rb") as f:
            _ = pickle.load(f)  # only to ensure file exists

        from PhysTwin.qqtt.data import RealData

        self.dataset = RealData(visualize=False, save_gt=False)
        self.object_points = self.dataset.object_points
        self.controller_points = self.dataset.controller_points
        self.structure_points = self.dataset.structure_points
        self.object_colors = self.dataset.object_colors
        self.object_visibilities = self.dataset.object_visibilities
        self.object_motions_valid = self.dataset.object_motions_valid
        self.num_original_points = self.dataset.num_original_points
        self.num_surface_points = self.dataset.num_surface_points
        self.num_all_points = self.dataset.num_all_points

        self.init_masks = None
        self.init_velocities = None

        # Use trainer logic to generate spring structure
        tmp = InvPhyTrainerWarp(
            data_path=self.data_path,
            base_dir=self.exp_base_dir,
            pure_inference_mode=True
        )
        (
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            num_object_springs
        ) = tmp._init_start(
            self.structure_points,
            self.controller_points[0],
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        # Build simulator
        self.sim = SpringMassSystemWarp(
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

    def _load_trained_model(self):
        paths = sorted(glob.glob(self.model_ckpt_glob))
        assert paths, f"No model found at: {self.model_ckpt_glob}"
        ckpt_path = paths[0]
        print(f"✅ Loading trained model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.sim.set_spring_Y(torch.log(checkpoint["spring_Y"]).detach().clone())
        self.sim.set_collide(
            checkpoint["collide_elas"].detach().clone(),
            checkpoint["collide_fric"].detach().clone(),
        )
        self.sim.set_collide_object(
            checkpoint["collide_object_elas"].detach().clone(),
            checkpoint["collide_object_fric"].detach().clone(),
        )

    def reset(self, init_idx=0, init_ctrl_pts=None, init_obj_pts=None):
        self.sim.reset()
        self.step_id = 0
        if init_ctrl_pts is None or init_obj_pts is None:
            init_ctrl_pts = self.controller_points[0]
            init_obj_pts = self.object_points[0]
        self.ctrl_pts = init_ctrl_pts.clone()
        self.gs_pts = init_obj_pts.clone()
        self.sim.set_init_state_from_numpy(self.ctrl_pts, self.gs_pts)
        return self.get_obs()

    def step(self, delta_ctrl):
        self.step_id += 1
        self.sim.step_ctrl(delta_ctrl)
        self.ctrl_pts = self.sim.get_control_points()
        self.gs_pts = self.sim.get_obj_pts()
        return self.get_obs()

    def get_obs(self):
        return {
            "ctrl_pts": self.ctrl_pts.clone().cpu(),
            "gs_pts": self.gs_pts.clone().cpu(),
        }

    def render(self):
        print(f"[Render] Step {self.step_id} | ctrl shape: {self.ctrl_pts.shape} | obj shape: {self.gs_pts.shape}")
