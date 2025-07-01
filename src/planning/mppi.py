# workspace/src/planning/mppi.py
import torch
import os
import pickle
import wandb
import numpy as np
from tqdm import tqdm
from src.planning.losses import chamfer

class MPPI:
    def __init__(self, env, horizon=50, num_samples=512, lambda_=1.0, noise_sigma=0.05, outer_iters=200,
             init_idx=0, target_idx=0, init_path=None):
        self.init_idx = init_idx
        self.target_idx = target_idx
        self.init_path = init_path
        self.env = env
        self.H = horizon
        self.N = num_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
        self.outer_iters = outer_iters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.u = torch.zeros((self.H, 30, 3), device=self.device)  # action sequence

    def rollout(self, actions):
        env = self.env.clone()  # You must implement clone() if possible
        env.reset_to_origin()
        rewards = 0.0
        for a in actions:
            obs, _, _, _ = env.step(a)
        final_ctrl = torch.tensor(env.get_ctrl_pts(), dtype=torch.float32).to(self.device)
        final_gs = torch.tensor(env.get_gs_pts(), dtype=torch.float32).to(self.device)
        return final_ctrl, final_gs

    def optimize(self, target_ctrl, target_gs):
        for outer_iter in tqdm(range(self.outer_iters), desc="MPPI outer iteration"):
            noise = torch.randn((self.N, self.H, 30, 3), device=self.device) * self.noise_sigma
            costs = torch.zeros((self.N,), device=self.device)

            for i in tqdm(range(self.N), desc=f"Outer {outer_iter} - Samples", leave=False):
                noisy_u = self.u + noise[i]
                self.env.reset_to_origin()
                max_step = 0.01
                for t in range(self.H):
                    clipped_action = torch.clamp(noisy_u[t], -max_step, max_step)
                    self.env.step(self.env.n_ctrl_parts, clipped_action)
                final_ctrl = torch.tensor(self.env.get_ctrl_pts()).to(self.device)
                final_gs = torch.tensor(self.env.get_gs_pts()).to(self.device)
                ctrl_loss = torch.nn.functional.mse_loss(final_ctrl, target_ctrl)
                chamfer_loss = chamfer(final_gs[None], target_gs[None])[0]
                costs[i] = ctrl_loss + chamfer_loss

            beta = torch.min(costs)
            weights = torch.exp(-(costs - beta) / self.lambda_)
            weights = weights / torch.sum(weights + 1e-10)
            weighted_noise = torch.sum(weights.view(-1, 1, 1, 1) * noise, dim=0)
            self.u = self.u + weighted_noise

            wandb.log({
                "mpc_outer_iter": outer_iter,
                "mean_cost": costs.mean().item(),
                "min_cost": costs.min().item(),
                "max_cost": costs.max().item(),
            })

            SAVE_ITERS = {1, 5, 20, 50, 80, 100, 150, 200, 400, 500}

            if outer_iter + 1 in SAVE_ITERS:
                self.env.set_init_state_from_numpy(self.init_path)
                ctrl_traj, gs_traj = [], []
                for t in range(self.u.shape[0]):
                    self.env.step(self.env.n_ctrl_parts, self.u[t])
                    ctrl_traj.append(self.env.get_obs()["ctrl_pts"])
                    gs_traj.append(self.env.get_obs()["state"])

                final_ctrl = torch.tensor(ctrl_traj[-1], dtype=torch.float32).to(self.device)
                final_gs = torch.tensor(gs_traj[-1], dtype=torch.float32).to(self.device)
                chamfer_loss = chamfer(final_gs[None], target_gs[None])[0]
                ctrl_loss = torch.nn.functional.mse_loss(final_ctrl, target_ctrl)

                # Save
                save_path = f"PhysTwin/mpc_output/init{self.init_idx:03d}_target{self.target_idx:03d}_iters{outer_iter+1}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "ctrl_traj": np.array(ctrl_traj),
                        "gs_traj": np.array(gs_traj),
                        "target_ctrl": target_ctrl.detach().cpu().numpy(),
                        "target_gs": target_gs.detach().cpu().numpy(),
                        "optimized_actions": self.u.detach().cpu().numpy(),
                    }, f)

                wandb.log({
                    f"chamfer_{outer_iter+1}": chamfer_loss.item(),
                    f"ctrl_mse_{outer_iter+1}": ctrl_loss.item(),
                })
                print(f"💾 Saved checkpoint traj at outer_iter={outer_iter+1} to {save_path}")

        return self.u
    