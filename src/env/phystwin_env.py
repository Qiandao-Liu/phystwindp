# /workspace/src/env/phystwin_env.py
import numpy as np
import warp as wp
import torch
import glob
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


class PhysTwinEnv(InvPhyTrainerWarp):
    """
    Loading training data from: ./data/different_types/double_lift_cloth_1/final_data.pkl
    Keys in final_data.pkl: 
    dict_keys(['controller_mask', 'controller_points', 'object_points', 
    'object_colors', 'object_visibilities', 'object_motions_valid', 'surface_points', 'interior_points'])
    Back to root: ./workspace/PhysTwin/
    """ 
    case_name = "double_lift_cloth_1"
    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

    data_path = f"../../PhysTwin/data/different_types/{case_name}/final_data.pkl"
    base_dir = f"../../PhysTwin/temp_experiments/{case_name}"
    optimal_params = f"../../PhysTwin/experiments_optimization/{case_name}/optimal_params.pkl"
    calibrate = f"../../PhysTwin/data/different_types/{case_name}/calibrate.pkl"
    metadata = f"../../PhysTwin/data/different_types/{case_name}/metadata.json"
    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    gaussians_path = f"../../PhysTwin/gaussian_output/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    def __init__(self, 
                 data_path=data_path,
                 base_dir=base_dir,
                 train_frame=100,
                 pure_inference_mode=True):
        super().__init__(
            data_path=data_path,
            base_dir=base_dir,
            train_frame=train_frame,
            pure_inference_mode=pure_inference_mode)
        
        timer = Timer()
        self.timer = timer
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

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

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
        print(f"[reset] Reset at time {self.timer.stop():.3f}s.")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()
        
        current_target = self.simulator.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(self.gaussians_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            masks_ctrl_pts = []
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                masks_ctrl_pts.append(torch.from_numpy(mask))

            # 用 cluster center 的 x 坐标判断左右
            center0 = np.mean(vis_controller_points[masks_ctrl_pts[0]], axis=0)
            center1 = np.mean(vis_controller_points[masks_ctrl_pts[1]], axis=0)

            if center0[0] > center1[0]:  # x 坐标大的是右边
                print("Switching the control parts")
                masks_ctrl_pts = [masks_ctrl_pts[1], masks_ctrl_pts[0]]
        else:
            masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]
                ).to("cuda")
                print(f"[DEBUG] Hand {i} cluster points shape: {target_points.shape}")
                print(f"[DEBUG] Hand {i} cluster points: {target_points}")
                hand_positions.append(self._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self._find_closest_point(target_points)


    def step(self, action):
        print()



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
