# /workspace/src/planning/gradient_mpc.py
import torch
import warp as wp
from tqdm import trange
import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PhysTwin")))

from src.env.phystwin_env import PhysTwinEnv

def main(case_name="double_lift_cloth_1", init_idx=0, target_idx=0):
    """
    Step 1. Init Env by PhysTwinEnv
    """
    init_path = f"PhysTwin/mpc_init/init_{init_idx:03d}.pkl"
    target_path = f"PhysTwin/mpc_target_U/target_{target_idx:03d}.pkl"

    with open(init_path, "rb") as f:
        init_data = pickle.load(f)
    with open(target_path, "rb") as f:
        target_data = pickle.load(f)

    env = PhysTwinEnv(case_name)
    sim = env.simulator  # SpringMassSystemWarp

    # Set init_state
    init_ctrl_wp, init_obj_wp, target_ctrl_wp, target_obj_wp = setup_simulator_state(sim, init_data, target_data)

    """
    Run MPC
    """
    best_actions_wp = run_gradient_mpc(sim, init_ctrl_wp, init_obj_wp, target_ctrl_wp, target_obj_wp)
    best_actions_np = wp.to_torch(best_actions_wp).cpu().numpy()
    print("Optimized action sequence shape:", best_actions_np.shape)


def setup_simulator_state(sim, init_data, target_data):
    """
    用 Warp 操作将 init_data + target_data 填入 sim (SpringMassSystemWarp)
    """
    print("[setup_simulator_state] Initializing simulator")

    # (1) Init State & Speed
    wp_x_np = init_data["wp_x"]               # (P,3)
    wp_v_np = np.zeros_like(wp_x_np)          # speed=0

    wp_x = wp.array(wp_x_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
    wp_v = wp.array(wp_v_np, dtype=wp.vec3f, device="cuda", requires_grad=True)

    sim.set_init_state(wp_x, wp_v)

    # (2) Ctrl_pts
    ctrl_pts_np = init_data["ctrl_pts"]       # (Nc, 3)
    ctrl_pts_wp = wp.array(ctrl_pts_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
    sim.wp_original_control_point = wp.clone(ctrl_pts_wp, requires_grad=True)
    # sim.wp_target_control_point   = wp.clone(ctrl_pts_wp, requires_grad=True)

    # (3) Target obj_pts & Ctrl_pts
    tgt_ctrl_wp = wp.array(target_data["ctrl_pts"], dtype=wp.vec3f, device="cuda")
    tgt_obj_wp  = wp.array(target_data["object_points"], dtype=wp.vec3f, device="cuda")

    # (4) Spring-Mass
    spring_indices_np = init_data["spring_indices"]      # (Ns, 2)
    spring_rest_len_np = init_data["spring_rest_len"]    # (Ns,)

    spring_indices = wp.array(spring_indices_np, dtype=wp.vec2i, device="cuda")
    rest_lengths   = wp.array(spring_rest_len_np, dtype=float, device="cuda")

    sim.wp_springs = spring_indices
    sim.wp_rest_lengths = rest_lengths

    print(f"Done. Init ctrl_pts: {ctrl_pts_np.shape}, obj_pts: {wp_x_np.shape}, springs: {spring_indices_np.shape}")
    return ctrl_pts_wp, wp_x, tgt_ctrl_wp, tgt_obj_wp

# @wp.kernel
# def add_action_kernel(original: wp.array(dtype=wp.vec3f),
#                       action: wp.array(dtype=wp.vec3f),
#                       out: wp.array(dtype=wp.vec3f)):
#     tid = wp.tid()
#     out[tid] = original[tid] + action[tid]
@wp.kernel
def apply_action_to_target_kernel(base: wp.array(dtype=wp.vec3f),
                                   action: wp.array(dtype=wp.vec3f),
                                   out: wp.array(dtype=wp.vec3f)):
    tid = wp.tid()
    out[tid] = base[tid] + action[tid]
@wp.kernel
def mse_loss_kernel(pred: wp.array(dtype=wp.vec3f),
                    target: wp.array(dtype=wp.vec3f),
                    loss: wp.array(dtype=float)):
    tid = wp.tid()
    diff = pred[tid] - target[tid]
    wp.atomic_add(loss, 0, wp.dot(diff, diff))

def compute_loss_warp(sim, target_obj_wp):
    """
    用 Warp Kernel 计算 loss
    """
    pred = sim.wp_states[-1].wp_x  # curr_pts
    target = target_obj_wp         # target_pts

    loss = wp.zeros(1, dtype=float, device="cuda", requires_grad=True)

    wp.launch(
        mse_loss_kernel,
        dim=pred.shape,
        inputs=[pred, target],
        outputs=[loss],
    )
    
    return wp.clone(loss, requires_grad=True)  # warp array

def run_gradient_mpc(sim,
                     init_ctrl_wp: wp.array,
                     init_obj_wp: wp.array,
                     target_ctrl_wp: wp.array,
                     target_obj_wp: wp.array,
                     horizon=40, lr=1e-2, outer_iters=200):
    """
    Step 5. Gradient_based MPC by using Tape(), get variables from functions in Step 2.
    sim: SpringMassSystemWarp 实例
    """
    action_seq = wp.zeros((horizon, sim.num_control_points), dtype=wp.vec3f, requires_grad=True)

    for t in range(outer_iters):
        tape = wp.Tape()

        sim.set_init_state(init_obj_wp, wp.zeros_like(init_obj_wp))

        with tape:  
            ctrl_pts_wp = wp.clone(init_ctrl_wp, requires_grad=True) 

            for t in range(horizon):
                updated_ctrl = wp.zeros_like(ctrl_pts_wp, requires_grad=True)
                wp.launch(
                    apply_action_to_target_kernel,
                    dim=sim.num_control_points,
                    inputs=[ctrl_pts_wp, action_seq[t]],
                    outputs=[updated_ctrl],
                )

                sim.wp_target_control_point = updated_ctrl
                ctrl_pts_wp = updated_ctrl
                if sim.object_collision_flag:
                    sim.update_collision_graph()

                print(f"[Frame {t}] wp_target_control_point[0] =", wp.to_torch(sim.wp_target_control_point)[0])
                print(f"[Frame {t}] ctrl_pts_wp[0] = {wp.to_torch(ctrl_pts_wp)[0]}")
    
                sim.step()

            # Warp Chamfer Loss
            loss = compute_loss_warp(sim, target_obj_wp)
            print("LOSS TYPE =", type(loss))
            print("LOSS (Warp array) =", loss)
            print("LOSS (Torch tensor) =", wp.to_torch(loss))

            final_obj_torch = wp.to_torch(sim.wp_states[-1].wp_x)
            print("[DEBUG] final_obj_wp[0]:", final_obj_torch[0])
            action_seq_torch = wp.to_torch(action_seq)
            print("[DEBUG] action_seq[0][0]:", action_seq_torch[0][0])
            print("[TRACE] wp_target_control_point[0] =", wp.to_torch(sim.wp_target_control_point)[0])

        # Backward
        tape.backward(loss)
        print("[TRACE] grad check:", wp.to_torch(action_seq.grad)[0][0])

        grad = action_seq.grad
        grad_torch = wp.to_torch(grad)  # shape: (horizon, Nc, 3)
        print("grad norm =", grad_torch.norm().item())
        print("grad[0][0] =", grad_torch[0][0])

        action_seq = action_seq - lr * grad

        if t % 10 == 0:
            grad_torch = wp.to_torch(action_seq.grad)
            print("→ grad[0][0] =", grad_torch[0][0])

    return action_seq

if __name__ == "__main__":
    main(init_idx=0, target_idx=0)
