# workspace/PhysTwin/make_dp3_dataset.py
import os
import glob
import pickle
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import zarr
import sys
sys.path.append('/home/qiandaoliu/workspace/mujoco_gs/aloha_custom')
from traj_in_mujoco_offscreen import solve_ik, find_closest_ctrl

"""
python make_dp3_dataset.py \
  --scene ~/workspace/mujoco_gs/aloha_custom/scene_with_cloth.xml \
  --pkl_dir ~/workspace/PhysTwin/mpc_replay/ \
  --temp_dir ~/workspace/PhysTwin/dp3_data/temp_npz/ \
  --output_zarr ~/workspace/PhysTwin/dp3_data/cloth_aloha_dataset.zarr
"""

def fast_farthest_point_sampling(points, k):
    N = points.shape[0]
    centroids = np.zeros((k,), dtype=np.int32)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(k):
        centroids[i] = farthest
        centroid = points[farthest][None, :]
        dist = np.sum((points - centroid) ** 2, axis=-1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    return points[centroids]

def rollout_episode(scene_path, pkl_path, temp_dir, scale=1.0):
    import mujoco
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    with open(pkl_path, 'rb') as f:
        traj_data = pickle.load(f)
    if 'traj' in traj_data and 'gs_traj' not in traj_data:
        traj_data['gs_traj'] = traj_data['traj']

    traj = traj_data['gs_traj']
    ctrl_traj = traj_data['ctrl_traj']

    traj[..., 2] *= -1
    ctrl_traj[..., 2] *= -1

    R = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    traj = traj @ R.T
    ctrl_traj = ctrl_traj @ R.T

    num_frames, num_gs = traj.shape[0], traj.shape[1]

    all_points = np.concatenate([traj.reshape(-1, 3), ctrl_traj.reshape(-1, 3)], axis=0)
    center = all_points.mean(axis=0)

    table_top_z = -0.0009
    object_thickness = 0.041
    desired_base_z = table_top_z + object_thickness
    rope_offset = np.array([0.0, 0.0, desired_base_z])
    rope_scale = scale

    agent_poses, actions, point_clouds = [], [], []

    for t in tqdm(range(num_frames-1), desc=f"Rollout {os.path.basename(pkl_path)}"):
        for _ in range(50):
            mujoco.mj_forward(model, data)

        action = ctrl_traj[t+1] - ctrl_traj[t]
        action = action.reshape(-1)

        cloth = traj[t]
        # cloth_sampled = fast_farthest_point_sampling(cloth, 994)
        # 强行补全1024
        # If not enough points, pad with repeat
        if cloth.shape[0] < 994:
            pad_len = 994 - cloth.shape[0]
            pad_points = np.tile(cloth[-1:], (pad_len, 1))  # repeat last point
            cloth_padded = np.concatenate([cloth, pad_points], axis=0)
            cloth_sampled = cloth_padded
        else:
            cloth_sampled = fast_farthest_point_sampling(cloth, 994)
        ctrl = ctrl_traj[t]
        combined = np.concatenate([cloth_sampled, ctrl], axis=0)  # (1024, 3)
        combined = np.concatenate([combined, np.ones((combined.shape[0], 3))], axis=1)  # (1024, 6)
        assert combined.shape == (1024, 6), f"bad combined shape: {combined.shape}"

        agent_poses.append(ctrl_traj[t].reshape(-1))
        actions.append(action)
        point_clouds.append(combined)

    agent_poses = np.stack(agent_poses)
    actions = np.stack(actions)
    point_clouds = np.stack(point_clouds)

    basename = os.path.basename(pkl_path).replace('.pkl', '')
    np.savez(os.path.join(temp_dir, f'{basename}.npz'),
             agent_pos=agent_poses,
             action=actions,
             point_cloud=point_clouds)

def merge_all(temp_dir, output_path):
    files = sorted(glob.glob(os.path.join(temp_dir, '*.npz')))
    all_agent_pos, all_actions, all_pcs, meta_episode, meta_frame = [], [], [], [], []

    frame_idx = 0
    for ep_idx, file in enumerate(files):
        data = np.load(file)
        n = data['agent_pos'].shape[0]
        all_agent_pos.append(data['agent_pos'])
        all_actions.append(data['action'])
        all_pcs.append(data['point_cloud'])
        meta_episode.append(np.full((n,), ep_idx))
        meta_frame.append(np.arange(frame_idx, frame_idx + n))
        frame_idx += n

    final = {
        'agent_pos': np.concatenate(all_agent_pos, axis=0),
        'action': np.concatenate(all_actions, axis=0),
        'point_cloud': np.concatenate(all_pcs, axis=0),
        'meta_episode': np.concatenate(meta_episode, axis=0),
        'meta_frame': np.concatenate(meta_frame, axis=0)
    }

    root = zarr.open(output_path, mode='w')
    for key, value in final.items():
        root.create_dataset(name=key, data=value, chunks=True, dtype=value.dtype)

    print(f"✅ Saved final dataset to {output_path}, total {final['agent_pos'].shape[0]} frames.\n")

    # 打印dataset结构
    print("📦 Dataset structure:")
    for key, value in final.items():
        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--pkl_dir', type=str, required=True)
    parser.add_argument('--temp_dir', type=str, required=True)
    parser.add_argument('--output_zarr', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(args.pkl_dir, 'concat_*.pkl')))

    for pkl_file in pkl_files:
        rollout_episode(
            scene_path=args.scene,
            pkl_path=pkl_file,
            temp_dir=args.temp_dir
        )

    merge_all(args.temp_dir, args.output_zarr)

if __name__ == '__main__':
    main()
