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

"""
Step 1. Init Env by PhysTwinEnv
"""
env = PhysTwinEnv(case_name="double_lift_cloth_1")
sim = env.simulator  # SpringMassSystemWarp 的实例

# 假定你提前从文件里 load 好 init / target ctrl / obj 点
init_ctrl_pts = np.load(".../init_ctrl.npy")   # (Nc,3)
init_obj_pts  = np.load(".../init_obj.npy")    # (No,3)
target_ctrl   = np.load(".../tgt_ctrl.npy")    # (Nc,3)
target_obj    = np.load(".../tgt_obj.npy")     # (No,3)

"""
Step 2. Get all variables from SpringMassSystemWarp (This SpringMassSystemWarp should be the one we just activated by PhysTwin, not a new Dynamics Model)
    We don't use get_obs() from PhysTwinEnv to make sure it's Plain Warp + Tape
    虽然把变量从SpringMassSystemWarp里显式的拿出来没啥用, 但是我希望可以拿出来确定, 打印这些变量
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
Step 3. Small test that functions from SpringMassSystemWarp worked, like step()
"""
zero_action = wp.zeros(sim.num_control_points, dtype=wp.vec3)
# copy original->target so step does nothing
sim.set_controller_interactive(sim.wp_original_control_point, sim.wp_original_control_point)
if sim.object_collision_flag:
    sim.update_collision_graph()
sim.step()
print("After one sim.step():")
print(" wp_states[1].wp_x (first 5 verts):", wp.to_torch(sim.wp_states[1].wp_x)[:5])
# reset back to init
sim.set_init_state(sim.wp_states[0].wp_x, sim.wp_states[0].wp_v)


"""
Step 4. Define a Chamfer loss for both ctrl_pts and gs_pts
"""
def warp_chamfer(a_wp: wp.array, b_wp: wp.array) -> wp.array:
    """
    1) Warp -> Torch
    2) call your existing chamfer_torch
    3) torch scalar -> Warp scalar
    """
    a_t = wp.to_torch(a_wp, requires_grad=True)    # (M,3)
    b_t = wp.to_torch(b_wp, requires_grad=False)   # (N,3)
    # chamfer_torch expects batched: (1,M,3),(1,N,3)
    loss_t = chamfer(a_t.unsqueeze(0), b_t.unsqueeze(0)).mean()
    return wp.from_torch(loss_t, dtype=float)

"""
Step 5. Gradient_based MPC by using Tape(), get variables from functions in Step 2.
"""
def run_gradient_mpc(sim, 
                     init_ctrl_pts: np.ndarray,   # (Nc,3)
                     init_obj_pts:  np.ndarray,   # (No,3)
                     target_ctrl:   np.ndarray,   # (Nc,3)
                     target_obj:    np.ndarray,   # (No,3)
                     horizon=40, lr=1e-2, outer_iters=200):
    """
    sim: SpringMassSystemWarp 实例
    上面的 ctrl / obj 都是 numpy 或 Torch，用来初始化
    """
    # —— 1) 把初始点都转成 Warp 张量，并开启 requires_grad
    init_ctrl_wp = wp.from_torch(torch.tensor(init_ctrl_pts, dtype=torch.float32),
                                 dtype=wp.vec3, requires_grad=False)
    init_obj_wp  = wp.from_torch(torch.tensor(init_obj_pts,  dtype=torch.float32),
                                 dtype=wp.vec3, requires_grad=False)
    target_ctrl_wp = wp.from_torch(torch.tensor(target_ctrl, dtype=torch.float32),
                                   dtype=wp.vec3, requires_grad=False)
    target_obj_wp  = wp.from_torch(torch.tensor(target_obj,  dtype=torch.float32),
                                   dtype=wp.vec3, requires_grad=False)

    # —— 2) 令动作序列也是 Warp 张量（将优化它）
    #     shape = [horizon, Nc, 3]
    action_seq = wp.zeros((horizon, sim.num_control_points, 3), dtype=wp.vec3, requires_grad=True)

    # —— 3) 外层优化循环
    for it in range(outer_iters):
        # 重置 Warp Tape
        tape = wp.Tape()

        # 重置仿真到初始状态
        sim.set_init_state(init_obj_wp, wp.zeros_like(init_obj_wp))

        # 嵌入 Tape
        with tape:
            for t in range(horizon):
                # 直接把 action[t] 当作 target delta
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

            # —— 4) 计算 Warp 版的 Chamfer Loss
            loss_obj  = warp_chamfer(final_obj_wp,  target_obj_wp)
            loss_ctrl = warp_chamfer(final_ctrl_wp, target_ctrl_wp)
            loss      = loss_obj + loss_ctrl

        # —— 5) 反向传播、拿梯度、更新动作序列
        tape.backward(loss)

        # Warp 中拿到 gradients：
        grad = action_seq.grad  # 形状同 action_seq

        # 简单梯度下降
        action_seq = action_seq - lr * grad

        if it % 10 == 0:
            print(f"[iter {it:3d}] l_obj={wp.to_torch(loss_obj):.4f} "
                  f"l_ctrl={wp.to_torch(loss_ctrl):.4f}")
            
    return action_seq

"""
Step 6. Get the best actions from Step 5
"""
init_ctrl_pts = env.simulator.controller_points[0].cpu().numpy()
init_obj_pts  = wp.to_torch(env.simulator.wp_states[0].wp_x).cpu().numpy()
# 你应该自己准备好 target 数据
target_ctrl    = np.load(".../tgt_ctrl.npy")
target_obj     = np.load(".../tgt_obj.npy")

best_actions_wp = run_gradient_mpc(sim,
                                   init_ctrl_pts, init_obj_pts,
                                   target_ctrl,  target_obj,
                                   horizon=40, lr=1e-2, outer_iters=200)

best_actions = wp.to_torch(best_actions_wp).cpu().numpy()
print(">> optimized actions :", best_actions.shape)