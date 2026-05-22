import sys
# 为 stdout 和 stderr 使用行缓冲，确保日志能及时输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')

from omegaconf import OmegaConf
import numpy as np
import os
import pathlib
import hydra
import torch
import dill
from torch.utils.data import DataLoader

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

# =============== 【新增】科学计算库，用于处理旋转 ===============
from scipy.spatial.transform import Rotation

# 为了直接运行脚本，我们硬编码了参数
checkpoint = "data/outputs/2025.08.19/13.22.45_train_diffusion_unet_hybrid_lift_image/checkpoints/latest.ckpt"
device = 'cuda:0'
output_dir = "data/pusht_eval_output"


# =============== 【新增】辅助函数：将6D动作向量转为4x4变换矩阵 ===============
def action_6d_to_transform_matrix(action_6d):
    """
    将一个6D的动作向量 [tx, ty, tz, rx, ry, rz] 转换为4x4的齐次变换矩阵。
    旋转部分被假定为轴角表示法 (axis-angle)。
    """
    trans = action_6d[:3]
    rot_vec = action_6d[3:]
    
    # 创建一个4x4的单位矩阵
    matrix = np.eye(4)
    
    # 填充平移部分
    matrix[:3, 3] = trans
    
    # 处理旋转：只有当旋转向量非零时才进行转换
    if np.linalg.norm(rot_vec) > 1e-6:
        # 使用scipy从旋转向量创建旋转矩阵
        rotation = Rotation.from_rotvec(rot_vec)
        matrix[:3, :3] = rotation.as_matrix()
        
    return matrix
# ========================================================================


# =============== 逆归一化辅助函数 (无变化) ===============
def unnormalize_action(norm_action, min_vals, max_vals):
    original_action = min_vals + (norm_action + 1.0) / 2.0 * (max_vals - min_vals)
    return original_action
# =======================================================


def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        print(f"输出路径 {output_dir} 已存在，内容将被覆盖。")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()
    
    policy.set_normalizer(normalizer)
    device_obj = torch.device(device)
    policy.to(device_obj)
    policy.eval()

    global_min = torch.tensor([-8.815369, -12.179695, -3.694275], device=device_obj)
    global_max = torch.tensor([12.4626465, 6.562313, 27.29248], device=device_obj)

    print("\n开始验证 (计算单条轨迹的累积位姿误差)...")
    print("-" * 80)

    with torch.no_grad():
        # 只处理第一个批次的数据用于演示
        batch = next(iter(dataloader))
        
        # 将数据移动到指定设备
        batch = dict_apply(batch, lambda x: x.to(device_obj, non_blocking=True))
        obs_dict = batch['obs']
        gt_action_normalized = batch['action']

        # 使用策略模型进行预测
        result = policy.predict_action(obs_dict)
        pred_action_normalized = result['action_pred']
        
        # --- 逆归一化（与之前相同） ---
        gt_pos_normalized = gt_action_normalized[..., :3]
        gt_rot_normalized = gt_action_normalized[..., 3:]
        gt_pos_unnorm = unnormalize_action(gt_pos_normalized, global_min, global_max)
        gt_action_display = torch.cat([gt_pos_unnorm, gt_rot_normalized], dim=-1)

        pred_pos_normalized = pred_action_normalized[..., :3]
        pred_rot_normalized = pred_action_normalized[..., 3:]
        pred_pos_unnorm = unnormalize_action(pred_pos_normalized, global_min, global_max)
        pred_action_display = torch.cat([pred_pos_unnorm, pred_rot_normalized], dim=-1)

        # =============== 【核心修改】选择批次中的第一条轨迹进行处理 ===============
        # gt_trajectory 的形状是 (序列长度, 6)
        gt_trajectory = gt_action_display[0].cpu().numpy()
        pred_trajectory = pred_action_display[0].cpu().numpy()
        
        sequence_length = gt_trajectory.shape[0]

        # 初始化世界坐标系下的姿态矩阵
        gt_world_pose = np.eye(4)
        pred_world_pose = np.eye(4)
        
        print(f"正在分析第一条轨迹，序列长度为: {sequence_length} 步\n")

        # 遍历轨迹中的每一步
        for step_idx in range(sequence_length):
            # 取出当前步骤的6D动作，形状为 (6,)，这是正确的形状
            gt_action_6d = gt_trajectory[step_idx]
            pred_action_6d = pred_trajectory[step_idx]

            # 1. 将当前步骤的6D动作向量转换为4x4变换矩阵
            gt_step_matrix = action_6d_to_transform_matrix(gt_action_6d)
            pred_step_matrix = action_6d_to_transform_matrix(pred_action_6d)
            
            # 2. 通过矩阵乘法，累积（合成）姿态变换
            gt_world_pose = gt_world_pose @ gt_step_matrix
            pred_world_pose = pred_world_pose @ pred_step_matrix
            
            # 3. 计算真实姿态和预测姿态之间的误差变换
            error_matrix = np.linalg.inv(gt_world_pose) @ pred_world_pose
            
            # 4. 从误差矩阵中提取平移和旋转误差
            translation_error = np.linalg.norm(error_matrix[:3, 3])
            error_rotation = Rotation.from_matrix(error_matrix[:3, :3])
            rotation_error_deg = np.rad2deg(np.linalg.norm(error_rotation.as_rotvec()))
            
            print(f"--- 步骤 {step_idx + 1}/{sequence_length} ---")
            np.set_printoptions(precision=4, suppress=True)
            print(f"累积平移误差: {translation_error:.4f} (毫米/单位)")
            print(f"累积旋转误差: {rotation_error_deg:.4f} (度)\n")

    print("-" * 80)
    print("验证完成。")

if __name__ == '__main__':
    main(checkpoint, output_dir, device)