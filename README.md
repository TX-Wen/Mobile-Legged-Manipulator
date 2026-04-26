# Whole-Body Control of a Mobile Legged Manipulator
### 移动腿足机械臂的全身控制与多模态感知

![Status](https://img.shields.io/badge/Status-In_Preparation-blue)
![Target](https://img.shields.io/badge/Target-ICRA_/_IROS-success)
![Hardware](https://img.shields.io/badge/Hardware-Unitree_Go2_+_Arm-lightgrey)

> **Teaser:** An embodied deployment framework integrating low-level high-dynamic motion control (RL) and high-level multi-modal perception interfaces (Diffusion Policy) for a mobile legged manipulator.
> 
> *简介：一个为移动腿足机械臂设计的具身部署框架，融合了底层高动态运动控制（强化学习）与顶层多模态感知接口（扩散策略）。*

---

## 🤸‍♂️ High-Dynamic Locomotion
**(基于强化学习的高动态运动控制)**

在 Isaac Gym 环境中，通过“教师-学生”蒸馏架构与 PPO 强化学习算法，实现了机器人在复杂地形下的动作控制与高动态表现。

### 1. Backflip Control (后空翻控制)
<p align="center">
  <img src="assets/backflip.gif" width="60%" />
</p>

我们将训练逻辑分解为三个阶段，并设计了相应的事件触发奖励机制：
- **阶段 1：** 奖励保持稳定站立，为起跳蓄力奠定基础。
- **阶段 2：** 奖励俯仰角转速与累计翻转角度，同时惩罚滚转/偏航运动以抑制斜向翻转。
- **阶段 3：** 当翻转角度达到临界值时触发，奖励平稳着陆与姿态恢复（包括机身高度回归与足端稳定接触）。

---

## 🦾 Visuomotor Manipulation
**(基于视觉的精细操作与模仿学习)**

利用扩散策略（Diffusion Policy）构建了从原始视觉输入到电机指令的端到端映射，使机器人具备执行复杂交互任务的能力。

<p align="center">
  <img src="assets/elevator_task.gif" width="60%" />
</p>
<p align="center">
  <em>(执行“按电梯与开门”任务的模仿学习展示)</em>
</p>

- **Real2Sim & Sim2Real 闭环：** 利用真实世界的少样本数据校准仿真模型，精准复现物理与视觉特性，有效缩小了“真实-仿真”的数据分布差异。
- **自动化标定：** 实现了自动化的眼在手上（eye-in-hand）标定与数据对齐，确保了从仿真到现实的高保真迁移。

---

## ⚡ Asynchronous Smoothing Architecture
**(异步平滑架构)**

为了解决复杂交互中的实时性挑战，我们设计了一种异步解耦架构，显著提升了系统的响应速度：
- **低频推理层 (1Hz)：** 负责基于扩散模型的视觉感知与规划。
- **高频执行层 (50Hz+)：** 负责底层的电机动作追踪。

该架构将推理延迟从 2.0s 降低至 0.01s 以下，确保了机械臂在动态交互过程中的实时响应。

---

*(Note: The manuscript is currently in preparation for submission to ICRA/IROS. Full deployment code and pre-trained weights will be released upon publication. / 注：本论文正准备投递 ICRA/IROS，完整的部署代码和预训练权重将在论文发表后开源。)*
