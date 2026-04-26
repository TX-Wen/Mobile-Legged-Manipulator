# Whole-Body Control of a Mobile Legged Manipulator
### 移动腿足机械臂的全身控制与多模态感知

![Status](https://img.shields.io/badge/Status-In_Preparation-blue)
![Hardware](https://img.shields.io/badge/Hardware-Unitree_Go2_+_Arm-lightgrey)

> **Teaser:** An embodied deployment framework integrating low-level high-dynamic motion control (RL) and high-level multi-modal perception interfaces (Diffusion Policy) for a mobile legged manipulator.
> 
> *简介：一个为移动腿足机械臂设计的具身部署框架，融合了底层高动态运动控制（强化学习）与顶层多模态感知接口（扩散策略）。*

---

## 🤸‍♂️ High-Dynamic Locomotion (基于强化学习的高动态运动控制)

We trained highly dynamic skills and complex terrain traversal capabilities in Isaac Gym using a "Teacher-Student" distillation architecture and Proximal Policy Optimization (PPO).
[cite_start]*我们在 Isaac Gym 中，利用“教师-学生”蒸馏架构和 PPO 算法，训练了机器狗的高动态技能和穿越复杂地形的能力 [cite: 71]。*

### 1. Backflip Control (后空翻控制)
<p align="center">
  <img src="assets/backflip.gif" width="60%" />
</p>

[cite_start]The training process is formulated into three distinct stages with specific event-triggered reward mechanisms[cite: 47]:
[cite_start]*训练过程被分解为三个阶段，并设计了相应的事件触发与奖励机制 [cite: 47]：*
- [cite_start]**Stage 1 (Preparation):** Rewards stable standing to accumulate energy for the jump[cite: 48]. [cite_start]*(奖励保持稳定站立，为起跳蓄力奠定基础 [cite: 48]。)*
- [cite_start]**Stage 2 (Flipping):** Prioritizes pitch angular velocity and cumulative rotation angle, while penalizing roll/yaw movements to suppress diagonal flipping[cite: 49]. [cite_start]*(奖励以俯仰角转速和累计翻转角度为主，并惩罚滚转/偏航运动以抑制斜向翻转 [cite: 49]。)*
- **Stage 3 (Landing):** Triggered when the cumulative pitch angle reaches $\ge 270^\circ$. [cite_start]It rewards smooth landing and posture recovery (e.g., body height restoration, stable foot contact)[cite: 50]. [cite_start]*(当俯仰累计转角 $\ge 270^\circ$ 时触发，奖励平稳着陆与姿态恢复，包括机身高度回归和足端稳定接触 [cite: 50]。)*

---

## 🦾 Visuomotor Manipulation (基于视觉的精细操作与模仿学习)

Beyond locomotion, we built an end-to-end mapping from raw visual input to motor commands using Diffusion Policy for complex interaction tasks. 
*除了移动，我们利用扩散策略（Diffusion Policy）构建了从原始视觉输入到电机控制指令的端到端映射，用于执行复杂的交互任务。*

<p align="center">
  <img src="assets/elevator_task.gif" width="60%" />
</p>
<p align="center">
  [cite_start]<em>(Imitation learning for "pressing elevator buttons & opening doors" tasks / “按电梯 + 开门” 任务的模仿学习 [cite: 122])</em>
</p>

- [cite_start]**Real2Sim & Sim2Real Pipeline:** Bridging the reality gap by capturing real-world few-shot demonstrations to calibrate the simulation model, effectively aligning visual and physical characteristics[cite: 60, 61, 62]. [cite_start]*(通过真实少样本数据校准仿真模型，精准复现物理与视觉特性，缩小真实与仿真的数据分布差异 [cite: 60, 61, 62]。)*
- **Automated Calibration:** Enabled automated eye-in-hand calibration and temporal data alignment for seamless policy execution. *(实现了自动化的眼看手标定和时间数据对齐，确保策略的无缝执行。)*

---

## ⚡ Asynchronous Smoothing Architecture
**(异步平滑架构)**

*(注：这里可以放一张你之前提到的，展示高低频解耦的架构图)*

To ensure real-time responsiveness during complex interactions, we designed an asynchronous architecture that strictly decouples:
- **Low-frequency Reasoning (1Hz):** Diffusion-based visual planning.
- **High-frequency Execution (50Hz+):** Low-level motor tracking.

This paradigm drastically reduced inference latency from **2.0s to <0.01s**, unlocking the potential for highly reactive mobile manipulation.
*为了确保复杂交互的实时响应，我们设计了异步架构，将 1Hz 的扩散模型推理与 50Hz+ 的底层执行解耦，将推理延迟从 2秒骤降至 0.01秒以下。*
