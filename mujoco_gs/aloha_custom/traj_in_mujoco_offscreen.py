# traj_in_mujoco_offscreen.py
import os
import time
import pickle
import mujoco
import numpy as np
import imageio
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import argparse


'''
python ~/workspace/mujoco_gs/aloha_custom/traj_in_mujoco_offscreen.py \
  --scene ~/workspace/mujoco_gs/aloha_custom/scene_with_cloth.xml \
  --pkl ~/workspace/PhysTwin/mpc_replay/concat_commands0000.pkl \
  --fps 20 \
  --out_video replay_cloth.mp4 \
  --cloth \
  --scale 0.6
'''

def solve_ik(model, data, site_id, target_pos, steps=100, lr=0.05, tol=1e-5):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(steps):
        mujoco.mj_forward(model, data)
        site_pos = data.site_xpos[site_id]
        err = target_pos - site_pos
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        data.qpos += lr * jacp.T @ err
    mujoco.mj_forward(model, data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--pkl", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out_video", type=str, default="replay.mp4")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cloth", action="store_true", help="Use cloth ply color")
    return parser.parse_args()

# 帮助函数：找到一组ctrl points里最靠近cloth的那个点
def find_closest_ctrl(ctrl_points, cloth_points):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cloth_points)
    distances, indices = nbrs.kneighbors(ctrl_points)
    closest_idx = np.argmin(distances)
    return ctrl_points[closest_idx]

def main():
    args = parse_args()

    print(f"\U0001F4C2 Loading scene: {args.scene}")
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    print(f"\U0001F4C2 Loading trajectory: {args.pkl}")
    with open(args.pkl, "rb") as f:
        traj_data = pickle.load(f)
    if "traj" in traj_data and "gs_traj" not in traj_data:
        traj_data["gs_traj"] = traj_data["traj"]

    traj = traj_data["gs_traj"]
    ctrl_traj = traj_data["ctrl_traj"]
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

    all_points = np.concatenate([traj.reshape(-1, 3), ctrl_traj.reshape(-1, 3)], axis=0)
    center = all_points.mean(axis=0)
    
    table_top_z = -0.0009  # 桌子表面的 z
    object_thickness = 0.031  # 物体厚度，cloth很薄，可以小一点

    # min_z = traj[0,:,2].min()  # 第0帧 traj 的最小 z
    # print(f"🧹 traj第0帧最小z: {min_z:.4f}")

    desired_base_z = table_top_z + object_thickness
    z_offset = desired_base_z

    rope_offset = np.array([0.0, 0.0, z_offset])

    rope_scale = args.scale

    # === 加载颜色 ===
    data_path = "./data/different_types/double_lift_cloth_1"
    data_path = "./data/different_types/double_lift_cloth_3"

    if args.cloth:
        print(f"🎨 Loading color from: {data_path}/final_data.pkl")
        # 这里因为final_data.pkl里object_points和object_colors顺序对齐，所以可以直接拿来
        final_data_path = f"{data_path}/final_data.pkl"
        with open(final_data_path, 'rb') as f:
            final_data = pickle.load(f)
            object_points = final_data['object_points'][0]  # 取第0帧的点
            object_colors = final_data['object_colors'][0]  # 取第0帧的颜色！！
            object_colors = np.clip(object_colors, 0.0, 1.0)

        # 给traj里面每个gs点找到最邻近的object point, 拿到颜色
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(object_points)
        distances, indices = nbrs.kneighbors(traj[0])  # traj[0]: (num_gs,3)

        gs_colors = np.ones((traj.shape[1], 3))  # 默认白色
        valid_mask = indices[:,0] < len(object_colors)
        gs_colors[valid_mask] = object_colors[indices[valid_mask, 0]]
    else:
        print("fail color")

    for i, bid in enumerate(gs_body_ids):
        gid = model.geom(name=f"gs_{i:04d}").id
        rgba = np.append(gs_colors[i], 1.0)
        model.geom_rgba[gid] = rgba

    print(f"\U0001F3AC Start rendering {num_frames} frames...")

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

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(traj[0])

    # 初始IK
    # left_goal = (left_ctrl.mean(0) - center) * rope_scale + rope_offset
    # right_goal = (right_ctrl.mean(0) - center) * rope_scale + rope_offset

    # solve_ik(model, data, left_gripper_site_id, left_goal, steps=200, lr=0.1)
    # solve_ik(model, data, right_gripper_site_id, right_goal, steps=200, lr=0.1)
    # 初始化阶段用 traj[0]
    cloth_points_frame0 = traj[0]

    left_ctrl_point = find_closest_ctrl(left_ctrl, cloth_points_frame0)
    right_ctrl_point = find_closest_ctrl(right_ctrl, cloth_points_frame0)

    left_goal = (left_ctrl_point - center) * rope_scale + rope_offset
    right_goal = (right_ctrl_point - center) * rope_scale + rope_offset

    solve_ik(model, data, left_gripper_site_id, left_goal, steps=200, lr=0.1)
    solve_ik(model, data, right_gripper_site_id, right_goal, steps=200, lr=0.1)
    # === ===

    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0

    renderer = mujoco.Renderer(model, height=480, width=640)
    frame_imgs = []


    # === 自动设置输出视频路径 ===
    if args.out_video == "replay.mp4":
        basename = os.path.splitext(os.path.basename(args.pkl))[0]
        os.makedirs("replay_videos", exist_ok=True)
        args.out_video = os.path.join("replay_videos", f"{basename}.mp4")


    for t in range(num_frames):
        for i, bid in enumerate(gs_body_ids):
            pos = (traj[t, i] - center) * rope_scale + rope_offset
            model.body_pos[bid] = pos

        for i, bid in enumerate(ctrl_body_ids):
            pos = (ctrl_traj[t, i] - center) * rope_scale + rope_offset
            model.body_pos[bid] = pos

        # left_ctrl_frame = ctrl_traj[t][labels == left_label]
        # right_ctrl_frame = ctrl_traj[t][labels == right_label]

        # left_goal = (left_ctrl_frame.mean(0) - center) * rope_scale + rope_offset
        # right_goal = (right_ctrl_frame.mean(0) - center) * rope_scale + rope_offset
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

        frame_imgs.append(img)

    print(f"\U0001F3A5 Saving video to {args.out_video}...")
    imageio.mimsave(args.out_video, frame_imgs, fps=args.fps)
    print(f"✅ Done!")

if __name__ == "__main__":
    main()