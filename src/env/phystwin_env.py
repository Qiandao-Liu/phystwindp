# /workspace/src/env/phystwin_env.py
import numpy as np
import warp as wp
import torch
import glob
import os
import pickle
from PhysTwin.qqtt.engine.trainer_warp import InvPhyTrainerWarp
from PhysTwin.qqtt.utils import logger, cfg

from PhysTwin.gaussian_splatting.scene.gaussian_model import GaussianModel
from PhysTwin.gaussian_splatting.scene.cameras import Camera
from PhysTwin.gaussian_splatting.gaussian_renderer import render as render_gaussian
from PhysTwin.gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from PhysTwin.gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PhysTwin.gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from PhysTwin.gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion
from sklearn.cluster import KMeans


class PhysTwinEnv():
    """
    Loading training data from: ./data/different_types/double_lift_cloth_1/final_data.pkl
    Keys in final_data.pkl: 
    dict_keys(['controller_mask', 'controller_points', 'object_points', 
    'object_colors', 'object_visibilities', 'object_motions_valid', 'surface_points', 'interior_points'])
    Back to root: ./workspace/PhysTwin/
    """ 
    def __init__(self, 
                 case_name="double_lift_cloth_1",
                 train_frame=50,
                 pure_inference_mode=True,
                 ):
        self.case_name = case_name
        self.n_ctrl_parts = 2 

        # ===== 1. Set Path =====
        print("===== 1. Set Path =====")
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PHYSTWIN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../PhysTwin"))
        BEST_MODEL_GLOB = os.path.join(CURRENT_DIR, "../../PhysTwin/experiments", case_name, "train", "best_*.pth")
        exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
        best_model_files = glob.glob(BEST_MODEL_GLOB)
        if not best_model_files:
            raise FileNotFoundError(f"No best_*.pth found at {BEST_MODEL_GLOB}")

        data_path = os.path.join(PHYSTWIN_DIR, "data", "different_types", case_name, "final_data.pkl")
        base_dir = os.path.join(PHYSTWIN_DIR, "temp_experiments", case_name)
        optimal_path = os.path.join(PHYSTWIN_DIR, "experiments_optimization", case_name, "optimal_params.pkl")
        calibrate = os.path.join(PHYSTWIN_DIR, "data", "different_types", case_name, "calibrate.pkl")
        metadata = os.path.join(PHYSTWIN_DIR, "data", "different_types", case_name, "metadata.json")

        self.best_model_path = best_model_files[0]

        self.gaussians_path = os.path.join(
            PHYSTWIN_DIR, "gaussian_output", case_name,
            "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0",
            "point_cloud", "iteration_10000", "point_cloud.ply"
        )

        # ===== 2. Load Config =====
        print("===== 2. Load Config =====")
        if "cloth" in self.case_name or "package" in self.case_name:
            cfg.load_from_yaml(os.path.join(PHYSTWIN_DIR, "configs", "cloth.yaml"))
        else:
            cfg.load_from_yaml(os.path.join(PHYSTWIN_DIR, "configs", "real.yaml"))  

        logger.info(f"Load optimal parameters from: {optimal_path}")
        assert os.path.exists(
            optimal_path
        ), f"{case_name}: Optimal parameters not found: {optimal_path}"
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

        # ===== 3. Init a Trainer Warp =====
        print("===== 3. Init a Trainer Warp =====")        
        cfg.device = torch.device("cuda:0")
        trainer = InvPhyTrainerWarp(
            data_path=data_path,
            base_dir=base_dir,
            pure_inference_mode=pure_inference_mode
        )
        print(f"🟡🟡 data_path: {data_path}")
        print(f"🟡🟡 base_dir: {base_dir}")
        print(f"🟡🟡 pire_inference_mode: {pure_inference_mode}")

        self.trainer = trainer
        self.simulator = trainer.simulator

        # ===== 4. Init Scenario =====
        print("===== 4. Init Scenario =====")
        timer = Timer()
        self.timer = timer

        self.prev_target = self.simulator.controller_points[0].clone()
        self.current_target = self.simulator.controller_points[0].clone()
        self.prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()
        self.masks_ctrl_pts = []

        print(f"🟡🟡 best_model_path: {self.best_model_path}")
        self.init_scenario(self.best_model_path)


    """
    Init the Gym-Style Env
    """
    def init_scenario(self, best_model_path):
        self.timer.start()
        
        # Load the model
        logger.info(f"Load model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        # print(f"[DEBUG] Simulator spring count: {self.simulator.n_springs}")
        # print(f"[DEBUG] Checkpoint spring_Y count: {spring_Y.shape[0]}")

        # assert (
        #     len(spring_Y) == self.simulator.n_springs
        # ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        print(f"[init_scenario] Done in {self.timer.stop():.3f}s.")

    """
    Reset gs_pts and ctrl_pts to the original state
    Reset and clean spring_mass system
    """
    def reset_to_origin(self, n_ctrl_parts=2):
        # print(f"[reset] Reset at time {self.timer.stop():.3f}s.")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        self.prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()
        
        self.current_target = self.simulator.controller_points[0]
        self.prev_target = self.current_target

        vis_controller_points = self.current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(self.gaussians_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                self.masks_ctrl_pts.append(torch.from_numpy(mask))

            # 用 cluster center 的 x 坐标判断左右
            center0 = np.mean(vis_controller_points[self.masks_ctrl_pts[0]], axis=0)
            center1 = np.mean(vis_controller_points[self.masks_ctrl_pts[1]], axis=0)

            if center0[0] > center1[0]:  # x 坐标大的是右边
                # print("Switching the control parts")
                self.masks_ctrl_pts = [self.masks_ctrl_pts[1], self.masks_ctrl_pts[0]]
        else:
            self.masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = self.masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]
                ).to("cuda")
                # print(f"[DEBUG] Hand {i} cluster points shape: {target_points.shape}")
                # print(f"[DEBUG] Hand {i} cluster points: {target_points}")
                hand_positions.append(self.trainer._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self.trainer._find_closest_point(target_points)


    def step(self, n_ctrl_parts, action):
        print()
        self.simulator.set_controller_interactive(self.prev_target, self.current_target)
        if self.simulator.object_collision_flag:
            self.simulator.update_collision_graph()
        wp.capture_launch(self.simulator.forward_graph)
        x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)

        # Set initial state for next step
        self.simulator.set_init_state(
            self.simulator.wp_states[-1].wp_x,
            self.simulator.wp_states[-1].wp_v,
        )

        torch.cuda.synchronize()

        self.prev_x = x.clone()

        self.prev_target = self.current_target
        """        
        ctrl_pts shape: [n_ctrl_parts, 3]
        """
        target_change = action
        if self.masks_ctrl_pts is not None:
            for i in range(n_ctrl_parts):
                if self.masks_ctrl_pts[i].sum() > 0:
                    self.current_target[self.masks_ctrl_pts[i]] += torch.tensor(
                        target_change[i], dtype=torch.float32, device=cfg.device
                    )
                    if i == 0:
                        self.hand_left_pos += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
                    if i == 1:
                        self.hand_right_pos += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
        else:
            self.current_target += torch.tensor(
                target_change, dtype=torch.float32, device=cfg.device
            )
            self.hand_left_pos += torch.tensor(
                target_change, dtype=torch.float32, device=cfg.device
            )

    def get_obs(self):
        ctrl_pts = self.current_target.clone().detach().cpu().numpy()
        state_pts = self.prev_x.detach().cpu().numpy()
        return {"ctrl_pts": ctrl_pts, "state": state_pts}
    
    def set_init_state_from_numpy(self, init_state_path):
        """
        Load .pkl file containing:
        - ctrl_pts: (N, 3)
        - gs_pts: (M, 3)
        - wp_x: (P, 3)
        - spring_indices: (Q, 2)
        - spring_rest_len: (Q,)
        """
        print(f"🔄 [set_init_state_from_numpy] Loading: {init_state_path}")
        with open(init_state_path, "rb") as f:
            data = pickle.load(f)

        # Set WP positions and velocities
        wp_x = wp.array(data["wp_x"], dtype=wp.vec3f, device="cuda")
        wp_v = wp.zeros_like(wp_x)

        self.simulator.set_init_state(wp_x, wp_v)
        self.prev_x = wp.clone(wp_x)

        # Set control points
        ctrl_pts = torch.tensor(data["ctrl_pts"], dtype=torch.float32, device=cfg.device)
        self.prev_target = ctrl_pts.clone()
        self.current_target = ctrl_pts.clone()
        self.simulator.set_controller_state(ctrl_pts)

        # Set spring connections
        spring_indices = torch.tensor(data["spring_indices"], dtype=torch.int32, device=cfg.device)
        rest_lengths = torch.tensor(data["spring_rest_len"], dtype=torch.float32, device=cfg.device)
        self.simulator.set_custom_springs(spring_indices, rest_lengths)

        # Re-cluster controller points
        self.reset_clusters(ctrl_pts.cpu().numpy())

        if self.masks_ctrl_pts is not None:
            hand_positions = []
            for i in range(self.n_ctrl_parts):
                target_points = ctrl_pts[self.masks_ctrl_pts[i]].to("cuda")
                # print(f"[DEBUG] Hand {i} cluster points shape: {target_points.shape}")
                # print(f"[DEBUG] Hand {i} cluster points: {target_points}")
                hand_positions.append(self.trainer._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            self.hand_left_pos = self.trainer._find_closest_point(ctrl_pts.to("cuda"))

        print(f"✅ [set_init_state_from_numpy] Done: ctrl_pts={ctrl_pts.shape}, wp_x={wp_x.shape}, springs={spring_indices.shape}")

    def reset_clusters(self, vis_controller_points, n_ctrl_parts=None):
        """
        Re-run clustering logic to assign masks_ctrl_pts.
        """
        if n_ctrl_parts is None:
            n_ctrl_parts = self.n_ctrl_parts

        self.masks_ctrl_pts = []
        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                self.masks_ctrl_pts.append(torch.from_numpy(mask))

            # Sort left/right by x
            center0 = np.mean(vis_controller_points[self.masks_ctrl_pts[0]], axis=0)
            center1 = np.mean(vis_controller_points[self.masks_ctrl_pts[1]], axis=0)
            if center0[0] > center1[0]:
                # print("Switching the control parts (left/right)")
                self.masks_ctrl_pts = [self.masks_ctrl_pts[1], self.masks_ctrl_pts[0]]

        else:
            self.masks_ctrl_pts = None
        self.mask_ctrl_pts = self.masks_ctrl_pts
        self.n_ctrl_parts = n_ctrl_parts

    def get_ctrl_pts(self):
        return self.simulator.get_controller_state()
    
    def get_gs_pts(self):
        return self.get_obs()["state"]

import time

class Timer:
    def __init__(self):
        self.reset()

    def start(self):
        self.t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t1 = time.time()
        return self.t1 - self.t0

    def reset(self):
        self.t0 = 0
        self.t1 = 0
