# /workspace/src/env/env_testment/test_warp_autodiff.py
import warp as wp
import torch
import numpy as np

# 必须先初始化
wp.init()
print(f"Warp Version: {wp.__version__}")

# 自定义一个 kernel：给每个粒子加控制输入力
@wp.kernel
def apply_control_force(pos: wp.array(dtype=wp.vec3),
                        vel: wp.array(dtype=wp.vec3),
                        ctrl: wp.array(dtype=wp.vec3),
                        dt: float):
    i = wp.tid()
    vel[i] += dt * ctrl[i]
    pos[i] += dt * vel[i]

# 初始化数据（2个粒子）
n_particles = 2
dt = 0.1
pos = wp.array([[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True)
vel = wp.array([[0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]], dtype=wp.vec3, requires_grad=True)
ctrl = wp.array([[0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True)

# 启动 Tape 记录操作
with wp.ScopedTape() as tape:
    tape.watch(pos)
    tape.watch(vel)
    tape.watch(ctrl)

    wp.launch(
        kernel=apply_control_force,
        dim=n_particles,
        inputs=[pos, vel, ctrl, dt],
        device=wp.get_preferred_device()
    )

    # 假设我们要最小化第 0 个粒子的位置距离目标 [0.0, 1.0, 0.0]
    loss_vec = pos.numpy()[0] - np.array([0.0, 1.0, 0.0])
    loss = torch.tensor(np.sum(loss_vec ** 2), dtype=torch.float32, requires_grad=True)

# 反向传播（会触发 warp 的 Tape 中的 backward）
loss.backward()

# 打印梯度
print("Gradient w.r.t. ctrl:")
print(tape.gradients[ctrl])
print("Gradient w.r.t. pos:")
print(tape.gradients[pos])
print("Gradient w.r.t. vel:")
print(tape.gradients[vel])
