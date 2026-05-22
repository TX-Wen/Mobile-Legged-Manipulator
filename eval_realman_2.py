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
# 假设您有一个基础的图像数据集类，否则可能需要调整此导入
# from diffusion_policy.dataset.base_image_dataset import BaseImageDataset 
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply

# 为了直接运行脚本，我们硬编码了参数
checkpoint = "data/outputs/2025.08.19/13.22.45_train_diffusion_unet_hybrid_lift_image/checkpoints/latest.ckpt"
device = 'cuda:0'
output_dir = "data/pusht_eval_output"

# =============== 逆归一化辅助函数 (此部分无变化) ===============
def unnormalize_action(norm_action, min_vals, max_vals):
    """
    将归一化到 [-1, 1] 的动作还原到原始尺度。
    :param norm_action: 归一化的动作张量
    :param min_vals: 原始动作的最小值张量
    :param max_vals: 原始动作的最大值张量
    :return: 原始尺度的动作张量
    """
    # 核心公式：从 [-1, 1] 映射回 [min, max]
    original_action = min_vals + (norm_action + 1.0) / 2.0 * (max_vals - min_vals)
    return original_action
# =======================================================

def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        # 在非交互式环境中，可以注释掉确认步骤，直接覆盖
        # click.confirm(f"输出路径 {output_dir} 已存在！是否覆盖？", abort=True)
        print(f"输出路径 {output_dir} 已存在，内容将被覆盖。")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载检查点
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 从工作空间获取策略模型
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

    # =============== 定义位置坐标的全局动作范围 ===============
    # min/max 值应只对应需要逆归一化的维度（即前3个维度）
    global_min = torch.tensor([-8.815369, -12.179695, -3.694275], device=device_obj)
    global_max = torch.tensor([12.4626465, 6.562313, 27.29248], device=device_obj)
    # =======================================================

    print("\n开始验证 (前3维为原始动作尺度, 后3维为归一化尺度)...")
    print("-" * 65)

    num_samples_to_validate = 10
    samples_validated = 0

    with torch.no_grad(): # 在评估模式下，关闭梯度计算以节省资源
        for batch in dataloader:
            if samples_validated >= num_samples_to_validate:
                break

            # 将数据移动到指定设备
            batch = dict_apply(batch, lambda x: x.to(device_obj, non_blocking=True))
            obs_dict = batch['obs']
            gt_action_normalized = batch['action'] # 这是完整的6维真实动作 (归一化)

            # 使用策略模型进行预测
            result = policy.predict_action(obs_dict)
            pred_action_normalized = result['action_pred'] # 这是完整的6维预测动作 (归一化)
            
            # =============== 【核心修改】处理6维动作，仅对前3维逆归一化 ===============
            # 1. 对真实动作 (Ground Truth) 进行处理
            gt_pos_normalized = gt_action_normalized[..., :3]     # 提取前3维位置
            gt_rot_normalized = gt_action_normalized[..., 3:]     # 提取后3维旋转
            gt_pos_unnorm = unnormalize_action(gt_pos_normalized, global_min, global_max) # 仅对位置逆归一化
            # 将逆归一化的位置和原始的旋转拼接回来，用于展示
            gt_action_display = torch.cat([gt_pos_unnorm, gt_rot_normalized], dim=-1)

            # 2. 对预测动作 (Predicted) 进行同样的处理
            pred_pos_normalized = pred_action_normalized[..., :3]
            pred_rot_normalized = pred_action_normalized[..., 3:]
            pred_pos_unnorm = unnormalize_action(pred_pos_normalized, global_min, global_max)
            # 拼接得到用于展示的预测动作
            pred_action_display = torch.cat([pred_pos_unnorm, pred_rot_normalized], dim=-1)
            # ====================================================================

            for i in range(len(gt_action_normalized)):
                if samples_validated >= num_samples_to_validate:
                    break

                # =============== 【修改】使用拼接后的6D混合尺度数据进行打印 ===============
                gt_display = gt_action_display[i].cpu().numpy()
                
                pred_display = pred_action_display[i].cpu().numpy()
                diff = pred_display - gt_display

                print(f"--- 样本 {samples_validated + 1} ---")
                np.set_printoptions(precision=4, suppress=True) # 设置numpy打印格式
                print(f" 真实动作 (GT):       {gt_display}")
                print(f" 预测动作 (Predicted): {pred_display}")
                print(f" 差值 (Pred - GT):   {diff}\n")
                
                samples_validated += 1

    print("-" * 65)
    print(f"验证完成。共显示 {samples_validated} 个样本。")


if __name__ == '__main__':
    main(checkpoint, output_dir, device)