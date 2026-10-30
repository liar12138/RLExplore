# mjlab 简单测试记录（2026-10-30）

本轮完成单环境接口、网络反向传播与确定性测试用例验证。**未启动 mjlab PPO 训练、实验扫描或批量评估。**

## 结果

| 项目 | 结果 |
|---|---|
| 仿真后端 | mjlab 1.5.0 / MuJoCo Warp 3.10.0.1，CUDA RTX 4090 |
| 配套环境 | MuJoCo 3.10.0、Warp 1.14.0、Torch 2.12.1、ONNX Runtime 1.23.2 |
| 机器人 | 指定 G1 Dex3 资产；43 个机器人自由度，SONIC 控制 29 个身体关节 |
| 单元与集成测试 | **10/10 通过**，最终运行约 15 秒 |
| GPU smoke 物理步 | **560** |
| 冻结 SONIC 推理 | **140 次** |
| RGB-D 与观察候选 | 成功生成，8 个候选 |
| Transformer 一次反向传播 | 有限梯度 |
| 代码编译与 shell 语法 | 通过 |

核心测试覆盖 SONIC 模式/排序/确定性、四类网络梯度、特权信息与 Actor 隔离、历史清空、padding、GAE 和遮挡未知语义。集成测试覆盖 Gym 观测契约、物体尺寸配对、GPU 碰撞边界，以及一个固定无遮挡场景中的真实释放、撤手与稳定支撑。

物理集成测试将 CPU `mujoco.mj_step()` 替换为报错函数，并确认实际 CUDA 步数，验证主后端确实通过 mjlab 执行。它是单用例回归测试，不能据此报告批量放置成功率。

## 本次修复

1. 将主后端接入 mjlab `Scene` 和 `Simulation`，控制、weld 释放和接触结果在 GPU 与主机读回之间同步；CPU 数据用于 IK、ONNX 输入与 EGL RGB-D 渲染。
2. 随机化物体尺寸和场景后重新编译 spec、上传模型，更新惯量、碰撞边界与静态 BVH，避免 MuJoCo 3.10 中只改尺寸而派生数据未更新。
3. 接触读取使用 MuJoCo Warp 正确回传的 `contact.geom` 配对。
4. 释放撤手后允许有界稳定时间：最多检查三个 0.8 秒窗口，每个窗口采样 8 次。成功要求完整窗口内始终有合法支撑且无机器人接触，最大位移小于 1 cm、最大转角小于 0.035 rad。瞬时线速度只记录为诊断值；稳定性以窗口内的位姿和接触为准，避免接触抖动误判。等待时间全部计入动作耗时。

## 复现命令与文件

```bash
cd /home/tcs/8t01/lhx/RLExplore
MJLAB_WARP_QUIET=1 ./scripts/python.sh -m unittest discover -s tests -v
MJLAB_WARP_QUIET=1 ./scripts/run.sh smoke --config configs/smoke.yaml
./scripts/run.sh doctor --set device=cuda:0
```

服务器文件：

- `runs/mjlab_tests.log`：10 项测试输出。
- `runs/mjlab_smoke/smoke.json`：GPU 步数、SONIC 调用、候选与梯度检查。
- `runs/mjlab_smoke/ego.png`：第一人称图像。
- `runs/mjlab_doctor.json`：运行时版本、资产路径与权重哈希。
- `runs/mjlab_validation.json`：本轮测试汇总。

旧 CPU MuJoCo 的日志保留，但不作为 mjlab 结果。当前仍是固定骨盆的局部任务，没有验证行走/房屋探索，没有训练收敛或多种子性能结论。
