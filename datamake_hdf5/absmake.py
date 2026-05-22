import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image

def create_robomimic_dataset(root_dir, output_path):
    """
    遍历根目录下的轨迹文件夹，并创建一个符合 Robomimic 结构的 HDF5 数据集。

    Args:
        root_dir (str): 包含以数字命名的轨迹文件夹的根目录。
        output_path (str): 生成的 HDF5 文件的保存路径。
    """
    # 查找所有以数字命名的轨迹文件夹
    traj_folders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()
    ])

    if not traj_folders:
        print(f"在 '{root_dir}' 目录下没有找到有效的轨迹文件夹。")
        return

    print(f"找到了 {len(traj_folders)} 条轨迹: {traj_folders}")

    total_samples = 0

    with h5py.File(output_path, 'w') as f:
        # 创建主 data 组
        data_grp = f.create_group("data")

        # 遍历每个轨迹文件夹
        for i, folder_name in enumerate(traj_folders):
            print(f"--- 正在处理轨迹: {folder_name} ---")
            traj_path = os.path.join(root_dir, folder_name)
            csv_path = os.path.join(traj_path, 'synchronized_robot_data.csv')
            images_dir = os.path.join(traj_path, 'images')

            # 检查必要的文件和文件夹是否存在
            if not os.path.exists(csv_path) or not os.path.exists(images_dir):
                print(f"警告: 在 {traj_path} 中缺少 'synchronized_robot_data.csv' 或 'images' 文件夹，已跳过。")
                continue

            # 1. 读取 CSV 数据
            df = pd.read_csv(csv_path)
            num_samples = len(df)
            if num_samples == 0:
                print(f"警告: {csv_path} 为空，已跳过。")
                continue
            
            total_samples += num_samples

            # 2. 创建轨迹组 (demo_0, demo_1, ...)
            demo_grp = data_grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = num_samples

            # 3. 提取并写入 actions (m11 到 m44)
            # ======================= 已更正的部分 =======================
            # 根据用户提供的正确列名手动定义 action_keys
            action_keys = [
                'm11', 'm12', 'm13', 'tx',
                'm21', 'm22', 'm23', 'ty',
                'm31', 'm32', 'm33', 'tz',
                'm41', 'm42', 'm43', 'm44'
            ]
            # ==========================================================
            
            actions = df[action_keys].to_numpy(dtype=np.float32)
            demo_grp.create_dataset("actions", data=actions)
            
            # 4. 创建 obs 组
            obs_grp = demo_grp.create_group("obs")

            # 5. 写入时间戳 [obs][time]
            timestamps = df["image_timestamp"].to_numpy(dtype=np.float64)
            obs_grp.create_dataset("time", data=timestamps)
            
            # 6. 读取并写入图像 [obs][robot0_eye_in_hand_image]
            image_list = []
            img_shape = None
            for idx, row in df.iterrows():
                # 注意: 您的CSV文件显示图片后缀是.jpg, 请确保这里和您的文件名一致
                img_filename = row["image_filename"].replace('.jpg', '.png') # 确保使用您实际的图片格式
                img_path = os.path.join(images_dir, img_filename)
                
                if not os.path.exists(img_path):
                    # 如果png不存在，尝试jpg
                    jpg_path = os.path.join(images_dir, row["image_filename"])
                    if os.path.exists(jpg_path):
                        img_path = jpg_path
                    else:
                        print(f"错误: 找不到图像文件 {img_path} 或 {jpg_path}!")
                        if img_shape is None:
                            print("错误：无法确定图像尺寸，跳过此轨迹。")
                            del data_grp[f"demo_{i}"]
                            total_samples -= num_samples
                            break
                        # 创建一个黑色图像作为占位符
                        img_array = np.zeros(img_shape, dtype=np.uint8)

                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        if img_shape is None:
                            img_shape = img_array.shape # 获取 H, W, C

                image_list.append(img_array)
            
            if not image_list: # 如果循环因为找不到第一张图而中断
                continue

            images_np = np.array(image_list, dtype=np.uint8)
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=images_np)

            # 7. (可选) 创建占位的 rewards 和 dones 数据集
            rewards = np.zeros(num_samples, dtype=np.float32)
            dones = np.zeros(num_samples, dtype=np.uint8)
            dones[-1] = 1 
            demo_grp.create_dataset("rewards", data=rewards)
            demo_grp.create_dataset("dones", data=dones)

            print(f"成功处理 {num_samples} 帧数据。图像尺寸: {img_shape}")

        # 写入样本总数属性
        data_grp.attrs["total"] = total_samples
        print(f"\n数据集创建完成！总样本数: {total_samples}")
        print(f"文件已保存至: {output_path}")

if __name__ == '__main__':
    # --- 配置参数 ---
    # 包含所有轨迹文件夹 (如 '3', '5') 的根目录
    TRAJECTORIES_ROOT_DIR = '/home/baishan/diffusion_policy_real/data/data' 
    # 输出的 HDF5 文件名
    OUTPUT_HDF5_FILE = 'my_robot_dataset.hdf5'

    create_robomimic_dataset(TRAJECTORIES_ROOT_DIR, OUTPUT_HDF5_FILE)