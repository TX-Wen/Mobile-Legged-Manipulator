import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
# 导入scipy中的Rotation模块
from scipy.spatial.transform import Rotation

def create_robomimic_dataset(root_dir, output_path):
    """
    遍历根目录下的轨迹文件夹，并创建一个符合 Robomimic 结构的 HDF5 数据集。

    Args:
        root_dir (str): 包含以数字命名的轨迹文件夹的根目录。
        output_path (str): 生成的 HDF5 文件的保存路径。
    """
    # 查找所有以数字命名的轨迹文件夹

    # 定义坐标系变换矩阵 T
    T = np.array([
        [-0.15441093,  0.85745271,  0.49084837, 13.37353295],
        [-0.36734396,  0.41136713, -0.83416755, -32.19565444],
        [-0.91717811, -0.30911477,  0.25146049, -48.37482606],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ], dtype=np.float32)

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
            # ======================= 计算 6D 相对动作 =======================
            action_keys = [
                'm11', 'm12', 'm13', 'tx', 'm21', 'm22', 'm23', 'ty',
                'm31', 'm32', 'm33', 'tz', 'm41', 'm42', 'm43', 'm44'
            ]
            absolute_poses_flat = df[action_keys].to_numpy(dtype=np.float32)

            relative_actions_6d = []
            for j in range(num_samples):
                action_6d = np.zeros(6, dtype=np.float32) # 初始化为零向量
                if j < num_samples - 1:
                    p_t = absolute_poses_flat[j].reshape(4, 4)
                    p_t_plus_1 = absolute_poses_flat[j + 1].reshape(4, 4)
                    p_t_transformed = p_t @ T
                    p_t_plus_1_transformed = p_t_plus_1 @ T
                    
                    try:
                        p_t_transformed_inv = np.linalg.inv(p_t_transformed)
                    except np.linalg.LinAlgError:
                        print(f"警告: 轨迹 {folder_name} 的第 {j} 帧矩阵是奇异的。使用单位矩阵代替。")
                        p_t_transformed_inv = np.identity(4, dtype=np.float32)

                    relative_action_matrix = p_t_transformed_inv @ p_t_plus_1_transformed

                    # 1. 提取平移和旋转部分
                    translation = relative_action_matrix[:3, 3]
                    rotation_matrix = relative_action_matrix[:3, :3]
                    
                    try:
                        # 2. 将旋转矩阵转换为欧拉角 (使用 'xyz' 顺序, 单位为弧度)
                        # Scipy 的 Rotation 可以处理临界情况 (如万向锁)
                        r = Rotation.from_matrix(rotation_matrix)
                        euler_angles = r.as_euler('xyz', degrees=False) # [rx, ry, rz]
                    except ValueError as e:
                        print(f"错误: 在轨迹 {folder_name} 的第 {j} 行，旋转矩阵转换失败: {e}")
                        # 转换失败时，将欧拉角设为零，防止程序中断
                        euler_angles = np.zeros(3, dtype=np.float32)

                    # 3. 合并为 6D 动作向量 [tx, ty, tz, rx, ry, rz]
                    action_6d = np.concatenate((translation, euler_angles))

                    # ======================= 新增 NaN 检查 =======================
                    # 检查 action_6d 中是否存在 NaN 值
                    if np.isnan(action_6d).any():
                        # 如果存在 NaN，打印警告信息，包含轨迹名和行号 (j)
                        print(f"警告: 在轨迹 '{folder_name}' 的第 {j} 行 (0-indexed) 检测到 NaN 动作。")
                        print(f"    - 计算出的动作为: {action_6d}")
                        # (可选) 你可以在这里决定如何处理，比如将 NaN 替换为0
                        # action_6d = np.nan_to_num(action_6d) 
                    # ==========================================================
                
                # 对于最后一帧, 动作保持为零向量 [0,0,0,0,0,0]
                relative_actions_6d.append(action_6d)
            
            # 将 6D 动作列表转换为Numpy数组并写入HDF5
            relative_actions_np = np.array(relative_actions_6d, dtype=np.float32)
            demo_grp.create_dataset("actions", data=relative_actions_np)

            # 4. 创建 obs 组
            obs_grp = demo_grp.create_group("obs")

            # 5. 写入时间戳 [obs][time]
            timestamps = df["image_timestamp"].to_numpy(dtype=np.float64)
            obs_grp.create_dataset("time", data=timestamps)
            
            # 6. 读取、处理并写入图像
            image_list = []
            img_shape = None
            for idx, row in df.iterrows():
                #img_filename = row["image_filename"]
                img_filename = row["image_filename"].replace('.jpg', '.png')
                img_path = os.path.join(images_dir, img_filename)
                
                img_array = None
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        # ======================= 新增图像处理逻辑 =======================
                        # 原始尺寸，例如 (1280, 720)
                        width, height = img.size

                        # 1. 中心裁剪为正方形
                        crop_size = min(width, height)
                        left = (width - crop_size) / 2
                        top = (height - crop_size) / 2
                        right = (width + crop_size) / 2
                        bottom = (height + crop_size) / 2
                        img_cropped = img.crop((left, top, right, bottom))

                        # 2. 缩放至目标尺寸
                        target_size = (288, 288)
                        # 使用 LANCZOS 滤波器进行高质量缩放
                        img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
                        
                        # 转换为Numpy数组
                        img_array = np.array(img_resized)
                        # ==========================================================
                else:
                    print(f"错误: 找不到图像文件 {img_path}!")
                    # 如果找不到图片，创建一个黑色占位符以避免程序崩溃
                    if img_shape is None:
                        # 如果是第一张图就找不到，我们只能假定一个形状
                        img_shape = (288, 288, 3)
                    img_array = np.zeros(img_shape, dtype=np.uint8)

                if img_shape is None:
                    img_shape = img_array.shape
                
                image_list.append(img_array)
            
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
    TRAJECTORIES_ROOT_DIR = '/home/baishan/diffusion_policy_real/data/data_wash' 
    # 输出的 HDF5 文件名
    OUTPUT_HDF5_FILE = 'my_robot_dataset_relative_6d_crop3.hdf5'

    create_robomimic_dataset(TRAJECTORIES_ROOT_DIR, OUTPUT_HDF5_FILE)