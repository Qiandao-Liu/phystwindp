# /workspace/src/env/phystwin_env.py
import numpy as np
import torch
import glob
from PhysTwin.qqtt.engine.trainer_warp import InvPhyTrainerWarp
from PhysTwin.qqtt.utils import cfg
import pickle, json, os

class PhysTwinEnv(InvPhyTrainerWarp):
    """
    Loading training data from: ./data/different_types/double_lift_cloth_1/final_data.pkl
    Keys in final_data.pkl: 
    dict_keys(['controller_mask', 'controller_points', 'object_points', 
    'object_colors', 'object_visibilities', 'object_motions_valid', 'surface_points', 'interior_points'])
    Back to root: ./workspace/PhysTwin/
    """ 
    data_path = "../../PhysTwin/data/different_types/double_lift_cloth_1/final_data.pkl"
    base_dir = "../../PhysTwin/temp_experiments/double_lift_cloth_1"
    optimal_params = "../../PhysTwin/experiments_optimization/double_lift_cloth_1/optimal_params.pkl"
    gaussian = "../../PhysTwin/gaussian_output/double_lift_cloth_1/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply"
    calibrate = "../../PhysTwin/data/different_types/double_lift_cloth_1/calibrate.pkl"
    metadata = "../../PhysTwin/data/different_types/double_lift_cloth_1/metadata.json"
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

    def step():

    def reset():

    