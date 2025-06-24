# workspace/src/planning/mppi.py
import torch
import numpy as np
from src.planning.losses import chamfer

class MPPI:
    def __init__(self, env, horizon=40, num_samples=512, lambda_=1.0, noise_sigma=0.05):
        self.env = env
        self.H = horizon
        self.N = num_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
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
        for _ in range(5):  # outer iterations
            noise = torch.randn((self.N, self.H, 30, 3), device=self.device) * self.noise_sigma
            costs = torch.zeros((self.N,), device=self.device)

            for i in range(self.N):
                noisy_u = self.u + noise[i]
                self.env.reset_to_origin()
                for t in range(self.H):
                    self.env.step(noisy_u[t])
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

        return self.u
