import sys
# 为 stdout 和 stderr 使用行缓冲，确保日志能及时输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')

from omegaconf import OmegaConf
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply

# 新增导入：我们需要 scipy 来处理旋转变换
# 如果尚未安装，请运行: pip install scipy
from scipy.spatial.transform import Rotation

# 为了直接运行脚本，我们硬编码了参数
checkpoint = "data/outputs/2025.08.19/13.22.45_train_diffusion_unet_hybrid_lift_image/checkpoints/latest.ckpt"
device = 'cuda:0'
output_dir = "data/pusht_eval_output"

# =============== 逆归一化辅助函数 (此部分无变化) ===============
def unnormalize_action(norm_action, min_vals, max_vals):
    """
    将归一化到 [-1, 1] 的动作还原到原始尺度。
    """
    original_action = min_vals + (norm_action + 1.0) / 2.0 * (max_vals - min_vals)
    return original_action

# =============== 6D动作与4x4变换矩阵的相互转换辅助函数 ===============
def action_6d_to_transform_matrix(action_6d: np.ndarray) -> np.ndarray:
    """
    将一个6D的动作向量 (3D平移 + 3D轴角旋转) 转换为4x4的齐次变换矩阵。
    """
    if action_6d.shape != (6,):
        raise ValueError(f"输入的6D动作向量形状必须是(6,), 但得到的是{action_6d.shape}")
    translation = action_6d[:3]
    rotation_vector = action_6d[3:]
    try:
        rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    except Exception:
        rotation_matrix = np.eye(3)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def transform_matrix_to_action_6d(matrix: np.ndarray) -> np.ndarray:
    """
    【新增】将一个4x4的齐次变换矩阵转换为6D动作向量 (3D平移 + 3D轴角旋转)。
    这是 action_6d_to_transform_matrix 的逆操作。
    """
    if matrix.shape != (4, 4):
        raise ValueError(f"输入的变换矩阵形状必须是(4, 4), 但得到的是{matrix.shape}")
    
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    try:
        rotation_vector = Rotation.from_matrix(rotation_matrix).as_rotvec()
    except Exception:
        rotation_vector = np.zeros(3)
    
    return np.concatenate([translation, rotation_vector])

# =================================================================

def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        print(f"输出路径 {output_dir} 已存在，内容将被覆盖。")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载检查点
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取策略模型
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # 实例化数据集和数据加载器
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()
    
    # 配置策略模型
    policy.set_normalizer(normalizer)
    device_obj = torch.device(device)
    policy.to(device_obj)
    policy.eval()

    global_min = torch.tensor([-8.815369, -12.179695, -3.694275], device=device_obj)
    global_max = torch.tensor([12.4626465, 6.562313, 27.29248], device=device_obj)

    print("\n开始验证 (每步输出瞬时动作差值和累积位姿误差)...")
    print("-" * 80)

    num_samples_to_validate = 10
    samples_validated = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_validated >= num_samples_to_validate:
                break

            batch = dict_apply(batch, lambda x: x.to(device_obj, non_blocking=True))
            obs_dict = batch['obs']
            gt_action_normalized = batch['action']
            result = policy.predict_action(obs_dict)
            pred_action_normalized = result['action_pred']
            
            gt_pos_unnorm = unnormalize_action(gt_action_normalized[..., :3], global_min, global_max)
            gt_action_display = torch.cat([gt_pos_unnorm, gt_action_normalized[..., 3:]], dim=-1)
            pred_pos_unnorm = unnormalize_action(pred_action_normalized[..., :3], global_min, global_max)
            pred_action_display = torch.cat([pred_pos_unnorm, pred_action_normalized[..., 3:]], dim=-1)

            for i in range(len(gt_action_normalized)):
                if samples_validated >= num_samples_to_validate:
                    break

                print(f"====== 样本 {samples_validated + 1} / {num_samples_to_validate} ======")
                np.set_printoptions(precision=4, suppress=True)

                gt_trajectory = gt_action_display[i].cpu().numpy()
                pred_trajectory = pred_action_display[i].cpu().numpy()
                
                sequence_length = gt_trajectory.shape[0]
                
                # 初始化世界坐标系下的姿态矩阵 (单位矩阵)
                gt_world_pose = np.eye(4)
                pred_world_pose = np.eye(4)
                
                # 【核心修改】遍历轨迹中的每一步，累积变换并计算当前步的累积误差
                for step_idx in range(sequence_length):
                    # --- 1. 获取当前步的动作 ---
                    gt_action_step = gt_trajectory[step_idx]
                    pred_action_step = pred_trajectory[step_idx]
                    
                    # --- 2. 累积姿态变换 ---
                    gt_step_matrix = action_6d_to_transform_matrix(gt_action_step)
                    pred_step_matrix = action_6d_to_transform_matrix(pred_action_step)
                    gt_world_pose = gt_world_pose @ gt_step_matrix
                    pred_world_pose = pred_world_pose @ pred_step_matrix
                    
                    # --- 3. 计算当前累积位姿的误差 ---
                    # 误差变换 T_error = T_gt⁻¹ @ T_pred
                    # 它描述了如何从当前的真实位姿变换到当前的预测位姿
                    gt_world_pose_inv = np.linalg.inv(gt_world_pose)
                    error_pose_matrix = gt_world_pose_inv @ pred_world_pose
                    
                    # 将误差矩阵转换回6D向量以便打印
                    cumulative_error_6d = transform_matrix_to_action_6d(error_pose_matrix)
                    
                    # --- 4. 打印当前步的信息 ---
                    step_diff = pred_action_step - gt_action_step
                    print(f"--- 步 {step_idx + 1}/{sequence_length} ---")
                    print(f"  瞬时动作差值: {step5_diff}")
                    print(f"  累积位姿误差: {cumulative_error_6d}")

                print(f"====== 样本 {samples_validated + 1} 分析完成 ======\n")
                samples_validated += 1

    print(f"验证完成。共分析 {samples_validated} 个样本。")


if __name__ == '__main__':
    main(checkpoint, output_dir, device)