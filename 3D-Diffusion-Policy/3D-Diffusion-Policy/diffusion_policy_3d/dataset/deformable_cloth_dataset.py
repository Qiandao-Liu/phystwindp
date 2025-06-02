# workspace/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/dataset/deformable_cloth_dataset.py
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
import numpy as np
import zarr

class DeformableClothDataset(BaseDataset):
    def __init__(self, zarr_path, horizon, pad_before, pad_after, seed=42, val_ratio=0.02, max_train_episodes=None):
        super().__init__()

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        root = zarr.open_group(zarr_path, mode='r')
        self.point_cloud = np.array(root["point_cloud"])  # [T, 1024, 4]
        self.agent_pos = np.array(root["agent_pos"])      # [T, 16]
        self.action = root["action"]            # [T, 90]
        self.meta_episode = root["meta_episode"][:]  # [T]

        # 构建合法的 sample 时间点 t
        self.samples = []
        for ep_id in np.unique(self.meta_episode):
            indices = np.where(self.meta_episode == ep_id)[0]
            ep_start, ep_end = indices[0], indices[-1] + 1
            for t in range(ep_start + pad_before, ep_end - horizon - pad_after):
                self.samples.append(t)

        self.samples = np.array(self.samples)
        np.random.seed(seed)
        np.random.shuffle(self.samples)

        split = int(len(self.samples) * (1 - val_ratio))
        self.train_indices = self.samples[:split]
        self.val_indices = self.samples[split:]

    def __len__(self):
        return len(self.train_indices)

    def __getitem__(self, idx):
        t0 = self.train_indices[idx]
        idxs = np.arange(t0 - self.pad_before, t0 + self.horizon + self.pad_after)

        pc_seq = np.stack([self.point_cloud[i][..., :3].astype(np.float32) for i in idxs], axis=0)    # [H + pad_before + pad_after, 1024, 3]
        ap_seq = np.stack([self.agent_pos[i].astype(np.float32) for i in idxs], axis=0)  # 不切 [:12]
        action_seq = self.action[idxs[:self.horizon]].astype(np.float32)

        return {
            "obs": {
                "point_cloud": pc_seq,  
                "agent_pos": ap_seq    
            },
            "action": action_seq        
        }

    def get_validation_dataset(self):
        val_set = DeformableClothDataset.__new__(DeformableClothDataset)
        val_set.__dict__ = self.__dict__.copy()
        val_set.train_indices = self.val_indices
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            # 'action': self.action[:],
            'action': self.action[:],
            'agent_pos': self.agent_pos[:],
            'point_cloud': self.point_cloud[..., :3]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer