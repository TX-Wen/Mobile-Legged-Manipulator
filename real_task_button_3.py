import torch
import dill  # dill 用于序列化更复杂 Python 对象
import hydra
from omegaconf import OmegaConf

# --- 配置区 ---
# 您训练好的模型检查点路径
CHECKPOINT_PATH = "/home/baishan/diffusion_policy_real/data/epoch=0050.ckpt"
# 训练时使用的原始数据集路径
DATASET_PATH = "/home/baishan/diffusion_policy_real/data/my_robot_dataset_normalized.hdf5"
# 您希望将 normalizer 保存到的路径
OUTPUT_PATH = "/home/baishan/diffusion_policy_real/data/normalizer.pkl"

def main():
    """
    该脚本加载一个模型检查点，实例化对应的数据集以获取 normalizer，
    然后将这个 normalizer 对象保存到一个独立文件中。
    """
    print(f"正在从模型检查点加载配置: {CHECKPOINT_PATH}")
    
    # 从检查点加载配置信息
    payload = torch.load(open(CHECKPOINT_PATH, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 重要提示：检查点中的数据集路径可能是旧的或相对路径。
    # 这里我们显式地覆盖它，确保加载正确的数据集来提取 normalizer。
    print(f"正在加载数据集以提取 normalizer: {DATASET_PATH}")
    cfg.task.dataset.dataset_path = DATASET_PATH

    # 使用配置来实例化数据集对象
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    
    # 从数据集中获取 normalizer
    normalizer = dataset.get_normalizer()
    print("成功获取 Normalizer。")
    
    # 使用 dill 将 normalizer 对象保存到文件
    print(f"正在将 Normalizer 保存到: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        dill.dump(normalizer, f)
        
    print("Normalizer 保存成功！")

if __name__ == "__main__":
    main()