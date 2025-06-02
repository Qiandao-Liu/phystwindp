import open3d as o3d
import numpy as np
import argparse
import os
from plyfile import PlyData

def read_gs_ply_with_colors(ply_path):
    plydata = PlyData.read(ply_path)
    verts = plydata['vertex'].data

    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=-1)
    colors = np.stack([verts['f_dc_0'], verts['f_dc_1'], verts['f_dc_2']], axis=-1)
    colors = np.clip(colors, 0, 1)  # 防止颜色越界

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def transfer_colors(object_ply_path, frame_ply_path, output_ply_path):
    obj_pcd = o3d.io.read_point_cloud(object_ply_path)
    frame_pcd = o3d.io.read_point_cloud(frame_ply_path)

    assert len(obj_pcd.points) == len(frame_pcd.points), "Point count mismatch!"

    # Use object.ply colors
    frame_pcd.colors = obj_pcd.colors

    o3d.io.write_point_cloud(output_ply_path, frame_pcd)
    print(f"✅ Saved color-transferred point cloud to {output_ply_path}")

def render_gs(frame_id):
    gs_dir = "./PhysTwin/mpc_targets"
    out_dir = "./PhysTwin/mpc_targets_colored"
    os.makedirs(out_dir, exist_ok=True)

    ctrl_path = os.path.join(gs_dir, f"{frame_id}.npy")
    ply_path = os.path.join(gs_dir, f"{frame_id}.ply")

    ctrl_points = np.load(ctrl_path)
    pcd = read_gs_ply_with_colors(ply_path)  # ✅ 用你自己的 parser

    # 控制点红色小球
    spheres = []
    for pt in ctrl_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(pt)
        sphere.paint_uniform_color([1, 0, 0])
        spheres.append(sphere)

    print(f"🔍 Rendering GS + control points for frame {frame_id}...")
    o3d.visualization.draw_geometries([pcd] + spheres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_id", type=str, help="Frame index like '0001'")
    args = parser.parse_args()

    render_gs(args.frame_id)
