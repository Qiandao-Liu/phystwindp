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
from src.planning.losses import chamfer

def main(case_name="double_lift_cloth", init_idx=0, target_idx=0):
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
    sim = env.simulator  # SpringMassSystemWarp 实例

    # 在 sim 里设置 init_state
    init_ctrl_wp, init_obj_wp, target_ctrl_wp, target_obj_wp = setup_simulator_state(sim, init_data, target_data)

    """
    Step 2. Get all variables from SpringMassSystemWarp (This SpringMassSystemWarp should be the one we just activated by PhysTwin, not a new Dynamics Model)
        We don't use get_obs() from PhysTwinEnv to make sure it's Plain Warp + Tape
    """
    print("=== SpringMassSystemWarp attributes ===")
    print("num_control_points:", sim.num_control_points)
    print("num_object_points: ", sim.num_object_points)
    # 原始控制点 & 目标控制点
    print("wp_original_control_point:", sim.wp_original_control_point.shape)
    print("wp_target_control_point:  ", sim.wp_target_control_point.shape)
    # 初始状态下的物体顶点
    print("wp_states[0].wp_x:", sim.wp_states[0].wp_x.shape)
    print("wp_states[0].wp_v:", sim.wp_states[0].wp_v.shape)

    """
    Run MPC
    """
    best_actions_wp = run_gradient_mpc(sim, init_ctrl_wp, init_obj_wp, target_ctrl_wp, target_obj_wp)
    best_actions_np = wp.to_torch(best_actions_wp).cpu().numpy()
    print("✅ Optimized action sequence shape:", best_actions_np.shape)


def setup_simulator_state(sim, init_data, target_data):
    """
    用 Warp 操作将 init_data + target_data 填入 sim (SpringMassSystemWarp)
    """
    print("🔧 [setup_simulator_state] Initializing simulator with pure Warp...")

    # (1) 初始位置和速度
    wp_x_np = init_data["wp_x"]               # (P,3)
    wp_v_np = np.zeros_like(wp_x_np)          # 初始速度为0

    wp_x = wp.array(wp_x_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
    wp_v = wp.array(wp_v_np, dtype=wp.vec3f, device="cuda", requires_grad=True)

    sim.set_init_state(wp_x, wp_v)

    # (2) 控制点
    ctrl_pts_np = init_data["ctrl_pts"]       # (Nc, 3)
    ctrl_pts_wp = wp.array(ctrl_pts_np, dtype=wp.vec3f, device="cuda", requires_grad=True)
    sim.wp_original_control_point = wp.clone(ctrl_pts_wp, requires_grad=True)
    sim.wp_target_control_point   = wp.clone(ctrl_pts_wp, requires_grad=True)

    # (3) 目标物体 & 控制点（用于 loss 比较）
    tgt_ctrl_wp = wp.array(target_data["ctrl_pts"], dtype=wp.vec3f, device="cuda")
    tgt_obj_wp  = wp.array(target_data["object_points"], dtype=wp.vec3f, device="cuda")

    # (4) 自定义弹簧结构（Spring 连接）
    spring_indices_np = init_data["spring_indices"]      # (Ns, 2)
    spring_rest_len_np = init_data["spring_rest_len"]    # (Ns,)

    spring_indices = wp.array(spring_indices_np, dtype=wp.int32, device="cuda")
    rest_lengths   = wp.array(spring_rest_len_np, dtype=float, device="cuda")

    sim.wp_springs = spring_indices
    sim.wp_rest_lengths = rest_lengths

    print(f"✅ Done. Init ctrl_pts: {ctrl_pts_np.shape}, obj_pts: {wp_x_np.shape}, springs: {spring_indices_np.shape}")
    return ctrl_pts_wp, wp_x, tgt_ctrl_wp, tgt_obj_wp


def warp_chamfer(a_wp: wp.array, b_wp: wp.array) -> wp.array:
    """
    Step 4. Define a Chamfer loss for both ctrl_pts and gs_pts
    1) Warp -> Torch
    2) call your existing chamfer_torch
    3) torch scalar -> Warp scalar
    """
    a_t = wp.to_torch(a_wp, requires_grad=True)    # (M,3)
    b_t = wp.to_torch(b_wp, requires_grad=False)   # (N,3)
    # chamfer_torch expects batched: (1,M,3),(1,N,3)
    loss_t = chamfer(a_t.unsqueeze(0), b_t.unsqueeze(0)).mean()
    return wp.from_torch(loss_t, dtype=float)


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
    # 令动作序列也是 Warp 张量（将优化它）
    action_seq = wp.zeros((horizon, sim.num_control_points, 3), dtype=wp.vec3, requires_grad=True)
    print("✅ action_seq.requires_grad =", action_seq.requires_grad)

    # 外层优化循环
    for it in range(outer_iters):
        # 重置 Warp Tape
        tape = wp.Tape()

        # 重置仿真到初始状态
        sim.set_init_state(init_obj_wp, wp.zeros_like(init_obj_wp))

        # 嵌入 Tape
        with tape:
            for t in range(horizon):
                # 直接把 action[t] 当作 target delta
                print(f"🔁 [iter {it}] Step {t}: action_seq[{t}][0] =", wp.to_torch(action_seq[t][0]))

                sim.set_controller_interactive(
                    sim.wp_original_control_point,
                    sim.wp_original_control_point + action_seq[t]
                )
                if sim.object_collision_flag:
                    sim.update_collision_graph()
                sim.step()

            # 到这一步，Sim.wp_states[-1].wp_x 就是仿真结束的 object 点
            final_obj_wp  = sim.wp_states[-1].wp_x
            final_ctrl_wp = sim.wp_states[-1].wp_control_x

            # 计算 Warp 版的 Chamfer Loss
            loss_obj  = warp_chamfer(final_obj_wp,  target_obj_wp)
            loss_ctrl = warp_chamfer(final_ctrl_wp, target_ctrl_wp)
            loss      = loss_obj + loss_ctrl

        # 反向传播、拿梯度、更新动作序列
        tape.backward(loss)
        grad_torch = wp.to_torch(grad)  # shape: (horizon, Nc, 3)
        print("✅ grad norm =", grad_torch.norm().item())
        print("✅ grad[0][0] =", grad_torch[0][0])

        # Warp 中拿到 gradients：
        grad = action_seq.grad  # 形状同 action_seq

        # 简单梯度下降
        action_seq = action_seq - lr * grad

        if it % 10 == 0:
            grad_torch = wp.to_torch(action_seq.grad)
            print(f"[iter {it:3d}] l_obj={wp.to_torch(loss_obj):.4f} "
                f"l_ctrl={wp.to_torch(loss_ctrl):.4f} | grad_norm={grad_torch.norm():.6f}")
            print("→ grad[0][0] =", grad_torch[0][0])

    return action_seq


if __name__ == "__main__":
    main(init_idx=0, target_idx=0)
