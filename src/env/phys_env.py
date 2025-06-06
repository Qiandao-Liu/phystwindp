# /workspace/src/env/gym_env.py

from PhysTwin.qqtt.engine.trainer_warp import InvPhyTrainerWarp

class PhysTwinEnv:
    def __init__(self, case_name):
        self.sim = InvPhyTrainerWarp(case_name)
    def reset(self, init_idx=None):
        self.sim.reset(init_idx)
        return self.sim.get_obs()
    def step(self, action):
        self.sim.step_with_action(action)
        obs = self.sim.get_obs()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info
    def render(self):
        return self.sim.render_current_frame()
