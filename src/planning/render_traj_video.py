# /workspace/src/planning/render_traj_video.py
"""
python src/planning/render_traj_video.py \
    --traj_path PhysTwin/mpc_output/init000_target000_iters1.pkl \
    --output_path src/videos/init000_target000_iters1.mp4
"""
import os
import pickle
import argparse
import numpy as np
import open3d as o3d
import imageio

def render_single_frame(gs_pts, ctrl_pts):
    gs_pcd = o3d.geometry.PointCloud()
    gs_pcd.points = o3d.utility.Vector3dVector(gs_pts)
    gs_pcd.paint_uniform_color([1, 0, 0])  # red for GS

    ctrl_pcd = o3d.geometry.PointCloud()
    ctrl_pcd.points = o3d.utility.Vector3dVector(ctrl_pts)
    ctrl_pcd.paint_uniform_color([0, 1, 0])  # green for ctrl

    return [gs_pcd, ctrl_pcd]

def main(traj_path, output_path="rendered_video.mp4"):
    with open(traj_path, "rb") as f:
        data = pickle.load(f)

    gs_traj = data["gs_traj"]  # (H, N, 3)
    ctrl_traj = data["ctrl_traj"]  # (H, 30, 3)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.get_render_option().point_size = 3.0

    images = []
    for t in range(len(gs_traj)):
        vis.clear_geometries()
        geometries = render_single_frame(gs_traj[t], ctrl_traj[t])
        for g in geometries:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        images.append(image_np)

    vis.destroy_window()
    print(f"📹 Rendering {len(images)} frames to {output_path} ...")
    imageio.mimsave(output_path, images, fps=10)
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, required=True, help="Path to .pkl trajectory file")
    parser.add_argument("--output_path", type=str, default="rendered_video.mp4", help="Output video path")
    args = parser.parse_args()
    main(args.traj_path, args.output_path)
