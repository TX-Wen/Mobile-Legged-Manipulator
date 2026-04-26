

https://github.com/user-attachments/assets/2503520f-d93d-4fca-8de9-a3fb53114afb

# Whole-Body Control of a Mobile Legged Manipulator
### 移动腿足机械臂的全身控制与多模态感知

![Status](https://img.shields.io/badge/Status-In_Preparation-blue)
![Hardware](https://img.shields.io/badge/Hardware-Unitree_Go2_+_Arm-lightgrey)

> **Teaser:** An embodied deployment framework integrating low-level high-dynamic motion control (RL) and high-level multi-modal perception interfaces (Diffusion Policy) for a mobile legged manipulator.
> 
> *简介：一个为移动腿足机械臂设计的具身部署框架，融合了底层高动态运动控制（强化学习）与顶层多模态感知接口（扩散策略）。*

---

## 🤸‍♂️ High-Dynamic Locomotion
**(基于强化学习的高动态运动控制)**

在 Isaac Gym 环境中，通过“教师-学生”蒸馏架构与 PPO 强化学习算法，实现了机器人在复杂地形下的动作控制与高动态表现。

### 1. Whole-Body Target Tracking (全身目标追踪控制)
Based on the [legged-robots-manipulation](https://github.com/aCodeDog/legged-robots-manipulation) framework, we implemented a robust whole-body controller for the Unitree Go2 + Arm. The policy jointly controls the quadrupedal base and the manipulator to accurately track moving 6D end-effector targets in real-time.

*基于 aCodeDog 的开源框架，我们在 Unitree Go2 + Arm 上实现了稳健的全身控制器。策略能够协同控制腿足底盘与机械臂，实现对 6D 末端执行器目标的实时、精准追踪。*

<p align="center">
  <video src="./target_3_x264.mp4" autoplay loop muted playsinline width="60%"></video>
</p>
<p align="center">
  <em>(RL-based whole-body control for end-effector target tracking / 基于强化学习的全身协同目标追踪)</em>
</p>

### 2. High-Dynamic Backflip (高动态后空翻控制)
<p align="center">
  <video src="./jump.mp4" autoplay loop muted playsinline width="60%"></video>
</p>

我们将后空翻的训练逻辑分解为三个阶段：
- **阶段 1 (蓄力)：** 奖励保持稳定站立，为起跳蓄力奠定基础。
- **阶段 2 (腾空)：** 奖励俯仰角转速与累计翻转角度，同时惩罚滚转/偏航运动以抑制斜向翻转。
- **阶段 3 (落地)：** 当翻转角度达到临界值时触发，奖励平稳着陆与机身姿态恢复。

---

## 🦾 Visuomotor Manipulation
**(基于视觉的精细操作与模仿学习)**

利用扩散策略（Diffusion Policy）构建了从原始视觉输入到电机指令的端到端映射，使机器人具备执行复杂交互任务的能力。

<p align="center">
  <video src="./button_real_1_x264.mp4" autoplay loop muted playsinline width="48%"></video>
  <video src="./button_inference.mp4" autoplay loop muted playsinline width="48%"></video>
</p>
<p align="center">
  <em>(Left: Real-world Execution / 左：真实环境按电梯执行画面 &nbsp;&nbsp;&nbsp; Right: Policy Inference / 右：扩散策略推理视角)</em>
</p>

---

## ⚡ Asynchronous Smoothing Architecture
**(异步平滑架构)**

为了解决复杂交互中的实时性挑战，我们设计了一种异步解耦架构，显著提升了系统的响应速度：
- **低频推理层 (1Hz)：** 负责基于扩散模型的视觉感知与规划。
- **高频执行层 (50Hz+)：** 负责底层的电机动作追踪。

该架构将推理延迟从 2.0s 降低至 0.01s 以下，确保了机械臂在动态交互过程中的实时响应。

---

https://github.com/user-attachments/assets/52eb571e-ea00-49fd-a008-51c5627e0dad

