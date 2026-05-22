# 文件名: eval_diffusion_transformer_hybrid_workspace.py (更新版)

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import copy
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_image_real_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

# =============== 新增：逆归一化辅助函数 ===============
def unnormalize_action(norm_action, min_vals, max_vals):
    """
    将归一化到 [-1, 1] 的动作还原到原始尺度。
    :param norm_action: 归一化的动作张量
    :param min_vals: 原始动作的最小值张量
    :param max_vals: 原始动作的最大值张量
    :return: 原始尺度的动作张量
    """
    # 核心公式
    original_action = min_vals + (norm_action + 1.0) / 2.0 * (max_vals - min_vals)
    return original_action
# =======================================================


class EvalDiffusionTransformerHybridWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        # ===== 新增代码 =====
        # 如果配置中使用了 EMA，我们也创建一个 EMA 模型来接收权重
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # ====================

    def run(self):
        cfg = self.cfg

        checkpoint_path = "baishan/diffusion_policy_real/data/outputs/2025.08.19/13.22.45_train_diffusion_unet_hybrid_lift_image/checkpoints/latest.ckpt"
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(ROOT_DIR, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"在路径 {checkpoint_path} 未找到检查点文件")

        print(f"正在从以下路径加载检查点: {checkpoint_path}")
        #self.load_checkpoint(path=checkpoint_path, exclude_keys=None, include_keys=None)
        self.load_checkpoint(path=checkpoint_path, exclude_keys=['optimizer'], include_keys=None)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        # ===== 修改/新增代码 =====
        # 选择要使用的策略：如果 EMA 模型存在，就用它
        policy = self.model
        if self.ema_model is not None:
            self.ema_model.to(device)
            policy = self.ema_model
        self.model.eval()

        # =============== 新增：定义全局动作范围 ===============
        # 使用您提供的数据，并转换为 PyTorch 张量
        global_min = torch.tensor([-8.815369, -12.179695, -3.694275], device=device)
        global_max = torch.tensor([12.4626465, 6.562313, 27.29248], device=device)
        # =======================================================

        print("\n开始验证 (输出为原始动作尺度)...")
        print("-" * 50)

        num_samples_to_validate = 10
        samples_validated = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_validated >= num_samples_to_validate:
                    break

                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = batch['obs']
                gt_action_normalized = batch['action']

                result = self.model.predict_action(obs_dict)
                pred_action_normalized = result['action_pred']
                
                # =============== 修改：对动作进行逆归一化 ===============
                # 假设您的动作维度是3。如果不是，您需要相应地调整 global_min/max。
                # 如果动作维度大于3，这里只逆归一化前3个维度。
                gt_action_unnorm = unnormalize_action(gt_action_normalized[..., :3], global_min, global_max)
                pred_action_unnorm = unnormalize_action(pred_action_normalized[..., :3], global_min, global_max)
                # =======================================================

                for i in range(len(gt_action_normalized)):
                    if samples_validated >= num_samples_to_validate:
                        break

                    # =============== 修改：使用逆归一化后的数据进行打印 ===============
                    gt_unnorm = gt_action_unnorm[i].cpu().numpy()
                    pred_unnorm = pred_action_unnorm[i].cpu().numpy()
                    diff = pred_unnorm - gt_unnorm

                    print(f"--- 样本 {samples_validated + 1} ---")
                    np.set_printoptions(precision=4, suppress=True)
                    print(f"  原始真实动作 (Ground Truth): {gt_unnorm}")
                    print(f"  原始预测动作 (Predicted):    {pred_unnorm}")
                    print(f"  差值 (预测值 - 真实值):        {diff}\n")
                    
                    samples_validated += 1

        print("-" * 50)
        print(f"验证完成。共显示 {samples_validated} 个样本。")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg):
    OmegaConf.resolve(cfg)
    workspace = EvalDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()