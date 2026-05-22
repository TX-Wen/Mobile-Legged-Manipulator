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
    新增功能：对 6D 动作的前三维（位移）进行全局归一化，映射到 [-1, 1] 区间。

    Args:
        root_dir (str): 包含以数字命名的轨迹文件夹的根目录。
        output_path (str): 生成的 HDF5 文件的保存路径。
    """
    # 定义坐标系变换矩阵 T
    T_old = np.array([
        [-0.15441093,  0.85745271,  0.49084837, 13.37353295],
        [-0.36734396,  0.41136713, -0.83416755, -32.19565444],
        [-0.91717811, -0.30911477,  0.25146049, -48.37482606],
        [ 0.          ,  0.          ,  0.          ,  1.          ]
    ], dtype=np.float32)
    T = np.array([[-3.77017613e-01, -3.36324805e-02, -9.25595255e-01,
         2.31622378e+01],
       [-2.70410618e-01, -9.51804296e-01,  1.44729680e-01,
        -1.81615755e+01],
       [-8.85853158e-01,  3.04856424e-01,  3.49752404e-01,
        -5.49158669e+01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]], dtype=np.float32)
    # 查找所有以数字命名的轨迹文件夹
    traj_folders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()
    ])

    if not traj_folders:
        print(f"在 '{root_dir}' 目录下没有找到有效的轨迹文件夹。")
        return

    print(f"找到了 {len(traj_folders)} 条轨迹: {traj_folders}")

    # 第一步：遍历所有数据，计算未归一化的动作并暂存，同时收集所有位移动作
    all_trajectory_data = []
    all_pos_actions = []

    for folder_name in traj_folders:
        print(f"--- 正在预处理轨迹: {folder_name} ---")
        traj_path = os.path.join(root_dir, folder_name)
        csv_path = os.path.join(traj_path, 'synchronized_robot_data.csv')
        images_dir = os.path.join(traj_path, 'images')

        if not os.path.exists(csv_path) or not os.path.exists(images_dir):
            print(f"警告: 在 {traj_path} 中缺少 'synchronized_robot_data.csv' 或 'images' 文件夹，已跳过。")
            continue

        df = pd.read_csv(csv_path)
        num_samples = len(df)
        if num_samples == 0:
            print(f"警告: {csv_path} 为空，已跳过。")
            continue

        action_keys = [
            'm11', 'm12', 'm13', 'tx', 'm21', 'm22', 'm23', 'ty',
            'm31', 'm32', 'm33', 'tz', 'm41', 'm42', 'm43', 'm44'
        ]
        absolute_poses_flat = df[action_keys].to_numpy(dtype=np.float32)

        relative_actions_6d = []
        for j in range(num_samples):
            action_6d = np.zeros(6, dtype=np.float32)
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
                translation = relative_action_matrix[:3, 3]
                rotation_matrix = relative_action_matrix[:3, :3]
                
                try:
                    r = Rotation.from_matrix(rotation_matrix)
                    euler_angles = r.as_euler('xyz', degrees=False)
                except ValueError as e:
                    print(f"错误: 在轨迹 {folder_name} 的第 {j} 行，旋转矩阵转换失败: {e}")
                    euler_angles = np.zeros(3, dtype=np.float32)

                action_6d = np.concatenate((translation, euler_angles))
                if np.isnan(action_6d).any():
                    print(f"警告: 在轨迹 '{folder_name}' 的第 {j} 行 (0-indexed) 检测到 NaN 动作。")
            
            relative_actions_6d.append(action_6d)
        
        relative_actions_np = np.array(relative_actions_6d, dtype=np.float32)
        
        all_pos_actions.append(relative_actions_np[:, :3])

        all_trajectory_data.append({
            "num_samples": num_samples,
            "actions": relative_actions_np,
            "timestamps": df["image_timestamp"].to_numpy(dtype=np.float64),
            "images_dir": images_dir,
            "image_filenames": df["image_filename"].str.replace('.jpg', '.png').tolist()
        })

    if not all_pos_actions:
        print("没有可处理的数据。")
        return

    # ======================= 计算全局位移 Min 和 Max =======================
    all_pos_actions_np = np.concatenate(all_pos_actions, axis=0)
    
    action_pos_min = np.min(all_pos_actions_np, axis=0)
    action_pos_max = np.max(all_pos_actions_np, axis=0)
    
    print("\n" + "="*50)
    print("全局位移动作统计信息：")
    print(f"  - 全局最小值 (min): {action_pos_min}")
    print(f"  - 全局最大值 (max): {action_pos_max}")
    print("="*50 + "\n")
    
    action_pos_range = action_pos_max - action_pos_min
    action_pos_range[action_pos_range == 0] = 1.0

    # 第二步：写入 HDF5 文件，应用归一化并存储元数据
    total_samples = 0
    with h5py.File(output_path, 'w') as f:
        data_grp = f.create_group("data")

        data_grp.attrs["action_pos_min"] = action_pos_min
        data_grp.attrs["action_pos_max"] = action_pos_max
        
        for i, traj_data in enumerate(all_trajectory_data):
            num_samples = traj_data["num_samples"]
            print(f"--- 正在写入轨迹: demo_{i} ---")
            
            total_samples += num_samples

            demo_grp = data_grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = num_samples

            # ======================= 归一化并写入动作 =======================
            unnormalized_actions = traj_data["actions"]
            
            pos_actions = unnormalized_actions[:, :3]
            
            ### 修正部分 ###
            # 应用新的归一化公式: 2 * (x - min) / (max - min) - 1
            # 这会将数据从 [min, max] 映射到 [-1, 1]
            normalized_pos_actions = 2.0 * (pos_actions - action_pos_min) / action_pos_range - 1.0
            
            normalized_actions = unnormalized_actions.copy()
            normalized_actions[:, :3] = normalized_pos_actions
            
            demo_grp.create_dataset("actions", data=normalized_actions)

            # ======================= 写入其他数据 =======================
            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset("time", data=traj_data["timestamps"])
            
            image_list = []
            img_shape = None
            for img_filename in traj_data["image_filenames"]:
                img_path = os.path.join(traj_data["images_dir"], img_filename)
                
                img_array = None
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        width, height = img.size
                        crop_size = min(width, height)
                        left = (width - crop_size) / 2
                        top = (height - crop_size) / 2
                        right = (width + crop_size) / 2
                        bottom = (height + crop_size) / 2
                        #img_cropped = img.crop((left, top, right, bottom))
                        target_size = (640, 360)
                        img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img_resized)
                else:
                    print(f"错误: 找不到图像文件 {img_path}!")
                    if img_shape is None:
                        img_shape = (640, 360, 3)
                    img_array = np.zeros(img_shape, dtype=np.uint8)

                if img_shape is None:
                    img_shape = img_array.shape
                
                image_list.append(img_array)
            
            images_np = np.array(image_list, dtype=np.uint8)
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=images_np)

            rewards = np.zeros(num_samples, dtype=np.float32)
            dones = np.zeros(num_samples, dtype=np.uint8)
            dones[-1] = 1
            demo_grp.create_dataset("rewards", data=rewards)
            demo_grp.create_dataset("dones", data=dones)
            print(f"图像原始尺寸: {img.shape}")
            print(f"成功处理 {num_samples} 帧数据。图像尺寸: {img_shape}")

        data_grp.attrs["total"] = total_samples
        print(f"\n数据集创建完成！总样本数: {total_samples}")
        print(f"文件已保存至: {output_path}")

if __name__ == '__main__':
    # --- 配置参数 ---
    TRAJECTORIES_ROOT_DIR = '/home/baishan/diffusion_policy_real/data/traj' 
    OUTPUT_HDF5_FILE = 'realman_dataset_normalized0.hdf5' # 建议改名以区分

    create_robomimic_dataset(TRAJECTORIES_ROOT_DIR, OUTPUT_HDF5_FILE)