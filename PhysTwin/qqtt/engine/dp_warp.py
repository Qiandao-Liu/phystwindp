# workspace/PhysTwin/qqtt/engine/dp_warp.py
from qqtt.engine.trainer_warp import InvPhyTrainerWarp
import numpy as np

class DPClothWarp(InvPhyTrainerWarp):
    def __init__(self, case_name, pure_inference_mode=True, **kwargs):
        data_path = f"PhysTwin/data/different_types/{case_name}/final_data.pkl"
        base_dir = f"PhysTwin/temp_experiments/{case_name}"

        super().__init__(data_path=data_path, base_dir=base_dir, pure_inference_mode=pure_inference_mode)
        self.max_steps = 100
        self.step_count = 0

    def reset(self):
        self.step_count = 0

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices,
            self.simulator.wp_init_velocities,
            pure_inference=True
        )

        # ✅ 从顶点获取布料点云
        gs_init = self.simulator.wp_init_vertices.numpy()
        ctrl_init = self.simulator.controller_points[0].cpu().numpy()

        print(f"[DEBUG] simulator type: {type(self.simulator)}")
        print(f"[DEBUG] hasattr(simulator, 'gaussians') = {hasattr(self.simulator, 'gaussians')}")
        print(f"[DP DEBUG] agent_pos shape: {ctrl_init.shape}, flatten: {ctrl_init.flatten().shape}")


        return {
            "point_cloud": gs_init.astype(np.float32),
            "agent_pos": ctrl_init.flatten().astype(np.float32)
        }


    def step(self, action):
        self.step_count += 1
        self.apply_action(action)
        gs_next, ctrl_next = self.get_current_state()
        obs = {
            "point_cloud": gs_next.astype(np.float32),
            "agent_pos": ctrl_next.flatten().astype(np.float32)
        }
        reward = 0.0
        done = self.step_count >= self.max_steps
        info = {
            "goal_achieved": False
        }
        return obs, reward, done, info
