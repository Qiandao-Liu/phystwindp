# workspace/src/planning/gradient_mpc_new.py
import torch
import warp as wp
from tqdm import trange
import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))
from src.env.phystwin_env import PhysTwinEnv
from src.planning.losses import chamfer

def main(case_name="double_lift_cloth_1", init_idx=0, target_idx=0):
    # Init an Environment by PhysTwinEnv
    env = PhysTwinEnv(case_name)
    sim = env.simulator  # sim = SpringMassSystemWarp

    """
    Init_000.pkl:
    {
    "ctrl_pts": (N, 3),
    "gs_pts": (N, 3),
    "wp_x": (N, 3),
    "spring_indices": (N, 2),
    "spring_rest_len": (N,)
    }

    Target_000.pkl:
    {
    "ctrl_pts": (N, 3),
    "gs_pts": (N, 3),
    "object_points": (N, 3)
    }
    """
    init_path = f"PhysTwin/mpc_init/init_{init_idx:03d}.pkl"
    target_path = f"PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"

    with open(init_path, "rb") as f:
        init_data = pickle.load(f)
    with open(target_path, "rb") as f:
        target_data = pickle.load(f)

    # Set init simulator state
    init_ctrl_wp, init_obj_wp, target_ctrl_wp, target_obj_wp = setup_simulator_state(sim, init_data, target_data)

    # Run MPC
    best_actions_wp = run_gradient_mpc(sim, init_ctrl_wp, init_obj_wp, target_ctrl_wp, target_obj_wp)
    best_actions_np = wp.to_torch(best_actions_wp).cpu().numpy()
    print("Optimized action sequence shape:", best_actions_np.shape)


def setup_simulator_state(sim, init_data, target_data):
    """
    Fill init_data and target_data into simulator SpringMassSystemWarp
    """
    # (1) Set init State & Speed
    wp_x_np = init_data["wp_x"]
    wp_v_np = np.zeros_like(wp_x_np)  # speed=0

    wp_x = wp.array(wp_x_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
    wp_v = wp.array(wp_v_np, dtype=wp.vec3f, device="cuda", requires_grad=True)

    sim.set_init_state(wp_x, wp_v, pure_inference=True)

    # (2) Set ctrl_pts
    ctrl_pts_np = init_data["ctrl_pts"]
    ctrl_pts_wp = wp.array(ctrl_pts_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
    sim.wp_original_control_point = wp.clone(ctrl_pts_wp, requires_grad=True)
    sim.wp_target_control_point   = wp.clone(ctrl_pts_wp, requires_grad=True)

    # (3) Set target obj_pts & Ctrl_pts
    tgt_ctrl_wp = wp.array(target_data["ctrl_pts"], dtype=wp.vec3f, device="cuda")
    tgt_obj_wp  = wp.array(target_data["object_points"], dtype=wp.vec3f, device="cuda")

    # (4) Set Spring-Mass
    spring_indices_np = init_data["spring_indices"]
    spring_rest_len_np = init_data["spring_rest_len"]

    spring_indices = wp.array(spring_indices_np, dtype=wp.vec2i, device="cuda")
    rest_lengths   = wp.array(spring_rest_len_np, dtype=float, device="cuda")

    sim.wp_springs = spring_indices
    sim.wp_rest_lengths = rest_lengths

    print(f"Done. Init ctrl_pts: {ctrl_pts_np.shape}, obj_pts: {wp_x_np.shape}, springs: {spring_indices_np.shape}")
    return ctrl_pts_wp, wp_x, tgt_ctrl_wp, tgt_obj_wp

@wp.kernel
def mse_loss_kernel(pred: wp.array(dtype=wp.vec3f),
                    target: wp.array(dtype=wp.vec3f),
                    loss_out: wp.array(dtype=float)):
    tid = wp.tid()
    diff = pred[tid] - target[tid]
    sq_dist = wp.dot(diff, diff)
    wp.atomic_add(loss_out, 0, sq_dist)

def compute_loss_warp(sim, target):
    # MSE
    pred = sim.wp_states[-1].wp_x
    assert pred.shape[0] == target.shape[0], f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    num_points = pred.shape[0]
    loss_out = wp.zeros(1, dtype=float, device="cuda", requires_grad=True)

    wp.launch(
        kernel=mse_loss_kernel,
        dim=num_points,
        inputs=[pred, target],
        outputs=[loss_out],
        device="cuda"
    )

    return loss_out, num_points

def forward(sim, ctrl_pts_wp, target_ctrl_wp, num_step):
    # Set controller
    sim.wp_target_control_point = ctrl_pts_wp

    for _ in range(num_step):
        sim.step()

    # final state
    return sim.wp_states[-1].wp_x

def run_gradient_mpc(sim, 
                     init_ctrl_wp, init_obj_wp, 
                     target_ctrl_wp, target_obj_wp,
                     num_step=40, 
                     lr=1e-2, 
                     iters=200):
    ctrl_pts_wp = wp.clone(init_ctrl_wp, requires_grad=True)

    for t in trange(iters):
        with tape:
            # Set init state to new state
            sim.set_init_state(init_obj_wp, wp.zeros_like(init_obj_wp))

            # Set controller
            sim.wp_target_control_point = ctrl_pts_wp

            # rollout
            for _ in range(num_step):
                sim.step()

            loss_out, num_points = compute_loss_warp(sim, target_obj_wp)
            loss = wp.array([wp.to_torch(loss_out)[0] / num_points], dtype=float, device="cuda", requires_grad=True)

        # backward
        tape.backward(loss)

        grad = tape.gradients[ctrl_pts_wp]
        if grad is None:
            print("GradNone")
        else:
            print("Grad")

        # gradient descent
        grad = tape.gradients[ctrl_pts_wp]
        ctrl_pts_wp -= lr * grad

        print(f"[Iter {t}] Loss: {loss:.4f}")

    return ctrl_pts_wp


if __name__ == "__main__":
    tape = wp.Tape()
    main(init_idx=0, target_idx=0)
