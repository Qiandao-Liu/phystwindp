# workspace/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/deformable_cloth_runner.py
import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.pytorch_util import dict_apply
from termcolor import cprint

from diffusion_policy_3d.env.deformable.deformable_cloth_env import DeformableClothEnv

class DeformableClothRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 case_name="double_lift_cloth_1",
                 eval_episodes=5,
                 max_steps=100,
                 tqdm_interval_sec=5.0,
                 fps=10,
                 crf=22,
                 render_size=84):
        super().__init__(output_dir)
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.fps = fps
        self.crf = crf
        self.render_size = render_size
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env = DeformableClothEnv(case_name=case_name)
        print("[DEBUG] DeformableClothRunner successfully loaded.")

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_success = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc="Eval in DeformableCloth Env", leave=False, mininterval=self.tqdm_interval_sec):
            obs = env.reset()
            policy.reset()
            done = False
            step_count = 0
            success = False

            while not done and step_count < self.max_steps:
                agent_pos_trimmed = obs['agent_pos'][:12]  # 保持和训练阶段一致
                obs_dict = {
                    'point_cloud': torch.from_numpy(obs['point_cloud']).to(device).unsqueeze(0),
                    'agent_pos': torch.from_numpy(agent_pos_trimmed).to(device).unsqueeze(0)
                }
                print("[Eval Debug] point_cloud shape:", obs_dict['point_cloud'].shape)
                print(f"[Eval Debug] agent_pos shape: {obs_dict['agent_pos'].shape}")
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action = action_dict['action'].squeeze(0).cpu().numpy()
                obs, reward, done, info = env.step(action)
                success = info.get("goal_achieved", False)
                step_count += 1

            all_success.append(success)

        # compute logs
        mean_success = float(np.mean(all_success))
        cprint(f"test_mean_score: {mean_success:.3f}", "green")

        log_data = {
            "test_mean_score": mean_success,
            "mean_success": mean_success,
        }

        return log_data
