# rollout_images_and_agentpos.py
import os
import glob
import pickle
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from tqdm import tqdm

from traj_in_mujoco_offscreen import solve_ik, find_closest_ctrl

"""
python ~/workspace/mujoco_gs/aloha_custom/rollout_images_and_agentpos.py \
  --scene ~/workspace/mujoco_gs/aloha_custom/scene_with_cloth.xml \
  --pkl_dir ~/workspace/PhysTwin/mpc_replay/ \
  --out_image_dir ~/workspace/PhysTwin/dp3_data/images/ \
  --out_agentpos_dir ~/workspace/PhysTwin/dp3_data/agent_pos/ \
  --fps 20
"""

def rollout_one_episode(scene_path, pkl_path, out_image_dir, out_agentpos_dir, fps=20, scale=0.6, cloth_color=True):
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
    num_ctrl = ctrl_traj.shape[1]

    gs_body_ids = [model.body(name=f"gs_{i:04d}_body").id for i in range(num_gs)]
    ctrl_body_ids = [model.body(name=f"ctrl_{i:04d}_body").id for i in range(num_ctrl)]
    left_gripper_site_id = model.site("left/gripper").id
    right_gripper_site_id = model.site("right/gripper").id

    # === 加载cloth颜色 ===
    if cloth_color:
        data_path = "./data/different_types/double_lift_cloth_1"
        print(f"🎨 Loading color from: {data_path}/final_data.pkl")
        final_data_path = os.path.join(data_path, 'final_data.pkl')
        with open(final_data_path, 'rb') as f:
            final_data = pickle.load(f)
            object_points = final_data['object_points'][0]  # 第0帧
            object_colors = final_data['object_colors'][0]  # 第0帧
            object_colors = np.clip(object_colors, 0.0, 1.0)

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(object_points)
        distances, indices = nbrs.kneighbors(traj[0])  # traj[0]是初始的gs点

        gs_colors = np.ones((traj.shape[1], 3))  # 默认白色
        valid_mask = indices[:, 0] < len(object_colors)
        gs_colors[valid_mask] = object_colors[indices[valid_mask, 0]]

        for i, bid in enumerate(gs_body_ids):
            gid = model.geom(name=f"gs_{i:04d}").id
            rgba = np.append(gs_colors[i], 1.0)
            model.geom_rgba[gid] = rgba
    else:
        print("⚪ Cloth color loading skipped.")

    all_points = np.concatenate([traj.reshape(-1, 3), ctrl_traj.reshape(-1, 3)], axis=0)
    center = all_points.mean(axis=0)

    table_top_z = -0.0009
    object_thickness = 0.041
    desired_base_z = table_top_z + object_thickness
    z_offset = desired_base_z
    rope_offset = np.array([0.0, 0.0, z_offset])

    rope_scale = scale

    # k-means初始化gripper
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(ctrl_traj[0])
    labels = kmeans.labels_
    group0 = ctrl_traj[0][labels == 0]
    group1 = ctrl_traj[0][labels == 1]
    if group0.mean(0)[0] < group1.mean(0)[0]:
        left_ctrl, right_ctrl = group0, group1
        left_label, right_label = 0, 1
    else:
        left_ctrl, right_ctrl = group1, group0
        left_label, right_label = 1, 0

    # solve initial IK
    cloth_points_frame0 = traj[0]
    left_ctrl_point = find_closest_ctrl(left_ctrl, cloth_points_frame0)
    right_ctrl_point = find_closest_ctrl(right_ctrl, cloth_points_frame0)

    left_goal = (left_ctrl_point - center) * rope_scale + rope_offset
    right_goal = (right_ctrl_point - center) * rope_scale + rope_offset

    solve_ik(model, data, left_gripper_site_id, left_goal, steps=200, lr=0.1)
    solve_ik(model, data, right_gripper_site_id, right_goal, steps=200, lr=0.1)

    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0

    renderer = mujoco.Renderer(model, height=480, width=640)

    basename = os.path.basename(pkl_path).split('.')[0]
    image_save_dir = os.path.join(out_image_dir, basename)
    agentpos_save_dir = os.path.join(out_agentpos_dir, basename)
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(agentpos_save_dir, exist_ok=True)

    for t in tqdm(range(num_frames), desc=f"Rollout {basename}"):
        for i, bid in enumerate(gs_body_ids):
            pos = (traj[t, i] - center) * rope_scale + rope_offset
            model.body_pos[bid] = pos

        for i, bid in enumerate(ctrl_body_ids):
            pos = (ctrl_traj[t, i] - center) * rope_scale + rope_offset
            model.body_pos[bid] = pos

        cloth_points_frame = traj[t]
        left_ctrl_frame = ctrl_traj[t][labels == left_label]
        right_ctrl_frame = ctrl_traj[t][labels == right_label]

        left_ctrl_point = find_closest_ctrl(left_ctrl_frame, cloth_points_frame)
        right_ctrl_point = find_closest_ctrl(right_ctrl_frame, cloth_points_frame)

        left_goal = (left_ctrl_point - center) * rope_scale + rope_offset
        right_goal = (right_ctrl_point - center) * rope_scale + rope_offset

        for _ in range(50):
            mujoco.mj_forward(model, data)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, left_gripper_site_id)
            data.qpos += 0.1 * jacp.T @ (left_goal - data.site_xpos[left_gripper_site_id])
            mujoco.mj_jacSite(model, data, jacp, jacr, right_gripper_site_id)
            data.qpos += 0.1 * jacp.T @ (right_goal - data.site_xpos[right_gripper_site_id])

        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        img = renderer.render()

        # Resize到(128,128)
        img_resized = cv2.resize(img, (128, 128))

        # Save image
        image_path = os.path.join(image_save_dir, f"frame_{t:04d}.png")
        cv2.imwrite(image_path, img_resized)

        # Save agent_pos
        agentpos_path = os.path.join(agentpos_save_dir, f"frame_{t:04d}.npy")
        np.save(agentpos_path, data.qpos.copy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--pkl_dir', type=str, required=True)
    parser.add_argument('--out_image_dir', type=str, required=True)
    parser.add_argument('--out_agentpos_dir', type=str, required=True)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--scale', type=float, default=0.6)
    parser.add_argument('--cloth_color', action='store_true', help='whether to load cloth color')
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.pkl_dir, 'concat_*.pkl')))

    for pkl_file in pkl_files:
        rollout_one_episode(
            scene_path=args.scene,
            pkl_path=pkl_file,
            out_image_dir=args.out_image_dir,
            out_agentpos_dir=args.out_agentpos_dir,
            fps=args.fps,
            scale=args.scale
        )


if __name__ == '__main__':
    main()
