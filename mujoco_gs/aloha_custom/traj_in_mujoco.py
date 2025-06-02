# traj_in_mujoco.py

import argparse
import pickle
import time
import numpy as np
import mujoco
import mujoco.viewer
import os
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
'''
python ~/workspace/mujoco_gs/aloha_custom/traj_in_mujoco.py \
  --scene ~/workspace/mujoco_gs/aloha_custom/scene_with_rope.xml \
  --pkl ~/workspace/PhysTwin/mpc_replay/concat_commands0000.pkl
'''
'''
python ~/workspace/mujoco_gs/aloha_custom/traj_in_mujoco.py \
  --scene ~/workspace/mujoco_gs/aloha_custom/scene_with_cloth.xml \
  --pkl ~/workspace/PhysTwin/mpc_replay/concat_commands0000.pkl
'''

def solve_ik(model, data, site_id, target_pos, steps=100, lr=0.05, tol=1e-5):
    """Gradient descent IK solver. Modifies data.qpos in-place."""
    for _ in range(steps):
        mujoco.mj_forward(model, data)
        site_pos = data.site_xpos[site_id]
        err = target_pos - site_pos
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        delta_qpos = lr * jacp.T @ err
        data.qpos += delta_qpos
    mujoco.mj_forward(model, data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Path to scene_with_rope.xml")
    parser.add_argument("--pkl", type=str, required=True, help="Path to trajectory .pkl file")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset_z", type=float, default=0.75)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"📂 Loading scene: {args.scene}")
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    print(f"📂 Loading trajectory: {args.pkl}")
    with open(args.pkl, "rb") as f:
        traj_data = pickle.load(f)      
    if "traj" in traj_data and "gs_traj" not in traj_data:
        traj_data["gs_traj"] = traj_data["traj"]

    traj = traj_data["gs_traj"]
    ctrl_traj = traj_data["ctrl_traj"]

    R = np.array([
        [ 0, -1, 0],
        [ 1,  0, 0],
        [ 0,  0, 1]
    ])

    traj = traj @ R.T
    ctrl_traj = ctrl_traj @ R.T

    print("GS traj min:", traj[0].min(0))
    print("GS traj max:", traj[0].max(0))
    print("CTRL traj min:", ctrl_traj[0].min(0))
    print("CTRL traj max:", ctrl_traj[0].max(0))

    num_frames, num_gs = traj.shape[0], traj.shape[1]
    num_ctrl = ctrl_traj.shape[1]

    gs_body_ids = [model.body(name=f"gs_{i:04d}_body").id for i in range(num_gs)]
    ctrl_body_ids = [model.body(name=f"ctrl_{i:04d}_body").id for i in range(num_ctrl)]
    left_gripper_site_id = model.site("left/gripper").id
    right_gripper_site_id = model.site("right/gripper").id

    all_points = np.concatenate([traj.reshape(-1, 3), ctrl_traj.reshape(-1, 3)], axis=0)
    center = all_points.mean(axis=0)
    print("📍 Center of rope before shift:", center)

    table_top_z = -0.0009
    rope_thickness = 0.02
    rope_offset = np.array([0.0, 0.0, table_top_z + rope_thickness + 0.1])
    rope_scale = args.scale
    custom_stretch = np.array([1, 1, 1])
    frame_duration = 1.0 / args.fps

    gs_ids = [model.geom(name=f"gs_{i:04d}").id for i in range(num_gs)]

    ply_path = os.path.expanduser("~/workspace/PhysTwin/data/different_types/rope_double_hand/shape/object.ply")
    print(f"🎨 Loading color from object.ply: {ply_path}")
    ply = PlyData.read(ply_path)
    vertex = ply['vertex'].data

    f_dc_0 = vertex['f_dc_0']
    f_dc_1 = vertex['f_dc_1']
    f_dc_2 = vertex['f_dc_2']
    gs_colors_raw = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=-1)
    gs_colors_normalized = (gs_colors_raw + 1.5) / 4.0
    gs_colors_normalized = np.clip(gs_colors_normalized, 0.0, 1.0)
    gs_colors = gs_colors_normalized[:traj.shape[1]]

    for i, gid in enumerate(gs_ids):
        rgba = np.append(gs_colors[i], 1.0)
        model.geom_rgba[gid] = rgba

    print(f"🎬 Starting playback: {num_frames} frames at {args.fps} FPS")
    print("First 5 gs points frame 0:", traj[0, :5])

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

    _, left_indices = nn.kneighbors([left_ctrl.mean(0)])
    _, right_indices = nn.kneighbors([right_ctrl.mean(0)])
    left_target = traj[0][left_indices[0]].mean(0)
    right_target = traj[0][right_indices[0]].mean(0)
    left_goal = (left_target - center) * rope_scale + rope_offset
    right_goal = (right_target - center) * rope_scale + rope_offset

    print("🎯 Running IK initialization...")
    solve_ik(model, data, left_gripper_site_id, left_goal, steps=200, lr=0.1)
    solve_ik(model, data, right_gripper_site_id, right_goal, steps=200, lr=0.1)
    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0
    print("✅ IK initialization done.")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for t in range(num_frames):
            start_time = time.time()

            if t == 0:
                rope_start = (traj[t, 0] - center) * custom_stretch * rope_scale + rope_offset
                rope_end = (traj[t, -1] - center) * custom_stretch * rope_scale + rope_offset
                rope_span = np.linalg.norm(rope_end - rope_start)
                print(f"📏 Distance between rope ends (frame 0): {rope_span:.4f} meters")

            for i, bid in enumerate(gs_body_ids):
                pos = (traj[t, i] - center) * custom_stretch * rope_scale + rope_offset
                model.body_pos[bid] = pos

            for i, bid in enumerate(ctrl_body_ids):
                pos = (ctrl_traj[t, i] - center) * custom_stretch * args.scale + rope_offset
                model.body_pos[bid] = pos

            left_ctrl_frame = ctrl_traj[t][labels == left_label]
            right_ctrl_frame = ctrl_traj[t][labels == right_label]
            _, left_indices = nn.kneighbors([left_ctrl_frame.mean(0)])
            _, right_indices = nn.kneighbors([right_ctrl_frame.mean(0)])
            left_target = traj[t][left_indices[0]].mean(0)
            right_target = traj[t][right_indices[0]].mean(0)
            left_goal = (left_target - center) * rope_scale + rope_offset
            right_goal = (right_target - center) * rope_scale + rope_offset

            for _ in range(10):
                mujoco.mj_forward(model, data)
                left_err = left_goal - data.site_xpos[left_gripper_site_id]
                right_err = right_goal - data.site_xpos[right_gripper_site_id]
                mujoco.mj_jacSite(model, data, jacp, jacr, left_gripper_site_id)
                data.qpos += 0.05 * jacp.T @ left_err
                mujoco.mj_jacSite(model, data, jacp, jacr, right_gripper_site_id)
                data.qpos += 0.05 * jacp.T @ right_err

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - start_time
            remaining = frame_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print("✅ Playback complete.")

if __name__ == "__main__":
    main()
