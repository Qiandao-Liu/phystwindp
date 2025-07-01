# test_warp_mpc_autodiff.py

import warp as wp
import numpy as np

wp.init()

@wp.kernel
def step_kernel(pos: wp.array(dtype=float),
                vel: wp.array(dtype=float),
                dt: float):
    i = wp.tid()
    pos[i] += dt * vel[i]


def run_mpc_autodiff():
    # === 设置 ===
    n_particles = 1
    dt = 0.1
    T = 10  # 时间步数
    target = 1.0

    # === 初始化 ===
    pos = wp.array(np.zeros(n_particles), dtype=float, requires_grad=True)
    vel = wp.array(np.full(n_particles, 0.5), dtype=float, requires_grad=True)  # 可优化参数

    # === 自动微分 Tape ===
    tape = wp.Tape()
    with tape:
        # 模拟 T 步移动
        for t in range(T):
            wp.launch(kernel=step_kernel,
                      dim=n_particles,
                      inputs=[pos, vel, dt],
                      device=wp.get_preferred_device())

        # === 计算损失（目标位置为 target）===
        # 由于 Warp 不支持直接在 Tape 内做 loss 运算，我们出 scope 后 numpy 操作
    loss = (pos.numpy()[0] - target) ** 2
    print(f"[INFO] Final Position: {pos.numpy()[0]:.4f}")
    print(f"[INFO] Loss: {loss:.6f}")

    # === 反向传播 ===
    tape.backward(grads={pos: wp.array(np.array([2.0 * (pos.numpy()[0] - target)]), dtype=float)})
    print(f"[INFO] ∇vel = {vel.grad.numpy()}")
    print(f"[INFO] ∇pos = {pos.grad.numpy()}")


if __name__ == "__main__":
    run_mpc_autodiff()
