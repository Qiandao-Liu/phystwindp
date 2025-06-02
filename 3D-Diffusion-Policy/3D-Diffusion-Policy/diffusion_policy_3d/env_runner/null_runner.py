from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.policy.base_policy import BasePolicy

class NullRunner(BaseRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BasePolicy):
        return {
            "test_mean_score": 0.0,
            "mean_success": 0.0,
        }
