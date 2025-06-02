# generate_scene_with_cloth_3.py
import os

"""
<mujoco model="aloha_scene">
  <compiler meshdir="assets" texturedir="assets"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4">
    <flag contact="disable"/>
  </option>
"""

def generate_scene_with_cloth_points(
    original_scene_path,
    output_scene_path,
    num_gs_points,
    num_ctrl_points=30,
    gs_radius=0.002,
    ctrl_radius=0.0001,
):
    with open(original_scene_path, "r") as f:
        xml_lines = f.readlines()

    insert_index = None
    for i, line in enumerate(xml_lines):
        if "</worldbody>" in line:
            insert_index = i
            break

    if insert_index is None:
        raise RuntimeError("❌ Could not find </worldbody> in the original scene XML.")

    new_lines = []

    # Insert Gaussian Splatting points
    new_lines.append('    <!-- Cloth GS points (as body-wrapped geom) -->\n')
    for i in range(num_gs_points):
        new_lines.append(f'    <body name="gs_{i:04d}_body" pos="0 0 0">\n')
        new_lines.append(
            f'      <geom name="gs_{i:04d}" type="sphere" size="{gs_radius}" '
            f'rgba="1 0 0 1" contype="0" conaffinity="0"/>\n'
        )
        new_lines.append('    </body>\n')

    # Insert Controller points
    new_lines.append('    <!-- Controller points -->\n')
    for i in range(num_ctrl_points):
        new_lines.append(f'    <body name="ctrl_{i:04d}_body" pos="0 0 0">\n')
        new_lines.append(
            f'      <geom name="ctrl_{i:04d}" type="sphere" size="{ctrl_radius}" '
            f'rgba="0 1 0 1" contype="0" conaffinity="0"/>\n'
        )
        new_lines.append('    </body>\n')

    # Inject
    new_xml = xml_lines[:insert_index] + new_lines + xml_lines[insert_index:]

    with open(output_scene_path, "w") as f:
        f.writelines(new_xml)

    print(f"✅ Wrote new cloth scene with {num_gs_points + num_ctrl_points} bodies to: {output_scene_path}")

if __name__ == "__main__":
    generate_scene_with_cloth_points(
        original_scene_path="./../mujoco_gs/aloha_custom/scene.xml",
        output_scene_path="./../mujoco_gs/aloha_custom/scene_with_cloth_3.xml",
        num_gs_points=7521,
        num_ctrl_points=30
    )
