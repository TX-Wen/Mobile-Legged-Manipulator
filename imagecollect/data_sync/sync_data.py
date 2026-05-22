import pandas as pd
import os

def synchronize_robot_data():
    """
    该函数用于加载机器人关节、末端位姿和图像时间戳数据，
    并根据最接近的时间戳将它们同步匹配。
    """
    # --- 文件路径定义 ---
    # 确保这些文件名与您本地的文件名完全一致
    image_file = '2.txt'
    joint_file = 'joint_angle_correct_timestamp.csv'
    pose_file = 'tracking_data_with_corrected_timestamp1.csv'
    output_file = 'synchronized_robot_data.csv'

    # --- 检查文件是否存在 ---
    for f in [image_file, joint_file, pose_file]:
        if not os.path.exists(f):
            print(f"错误：找不到文件 '{f}'。请确保代码和所有数据文件都在同一个目录下。")
            return

    # 1. 加载三个数据集
    print("正在加载数据文件...")
    image_timestamps_df = pd.read_csv(image_file)
    joint_angles_df = pd.read_csv(joint_file)
    tracking_pose_df = pd.read_csv(pose_file)
    print("数据加载完成。")

    # 2. 数据预处理和重命名
    print("正在预处理数据...")
    # --- 图像数据 ---
    # 根据帧序号创建图像文件名 (frame_index 0 -> color_frame_1.jpg)
    image_timestamps_df['image_filename'] = 'color_frame_' + (image_timestamps_df['frame_index'] + 1).astype(str) + '.jpg'
    image_timestamps_df = image_timestamps_df.rename(columns={'frame_timestamp': 'image_timestamp'})
    image_timestamps_df = image_timestamps_df.sort_values('image_timestamp').reset_index(drop=True)

    # --- 关节数据 ---
    joint_angles_df = joint_angles_df.rename(columns={'Corrected_Unix_Timestamp': 'joint_timestamp'})
    joint_angles_df = joint_angles_df.sort_values('joint_timestamp').reset_index(drop=True)

    # --- 位姿数据 ---
    tracking_pose_df = tracking_pose_df.rename(columns={'Original_Unix_Timestamp': 'pose_timestamp'})
    tracking_pose_df = tracking_pose_df.sort_values('pose_timestamp').reset_index(drop=True)
    print("数据预处理完成。")

    # 3. 使用 merge_asof 进行时间同步合并
    print("正在根据时间戳进行匹配...")
    # direction='nearest' 表示查找双向（之前或之后）最近的时间戳
    # 首先，合并图像和关节数据
    merged_df = pd.merge_asof(
        left=image_timestamps_df,
        right=joint_angles_df,
        left_on='image_timestamp',
        right_on='joint_timestamp',
        direction='nearest'
    )

    # 接着，将上面的结果与位姿数据合并
    final_merged_df = pd.merge_asof(
        left=merged_df,
        right=tracking_pose_df,
        left_on='image_timestamp',
        right_on='pose_timestamp',
        direction='nearest'
    )
    print("匹配完成。")

    # 4. 计算时间差以评估匹配质量
    final_merged_df['joint_time_diff_seconds'] = (final_merged_df['image_timestamp'] - final_merged_df['joint_timestamp']).abs()
    final_merged_df['pose_time_diff_seconds'] = (final_merged_df['image_timestamp'] - final_merged_df['pose_timestamp']).abs()

    # 5. 显示结果预览和统计
    print("\n--- 匹配结果预览 (前5行) ---")
    display_columns = [
        'image_filename', 'image_timestamp', 'joint_timestamp', 'joint_time_diff_seconds',
        'pose_timestamp', 'pose_time_diff_seconds', 'J1', 'J2', 'J3', 'tx', 'ty', 'tz'
    ]
    # 筛选出实际存在的列进行显示，避免因原始文件中缺少某些列而报错
    existing_display_columns = [col for col in display_columns if col in final_merged_df.columns]
    print(final_merged_df[existing_display_columns].head())

    print("\n--- 匹配时间差统计信息 ---")
    print(final_merged_df[['joint_time_diff_seconds', 'pose_time_diff_seconds']].describe())

    # 6. 保存结果到新的 CSV 文件
    try:
        final_merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n成功！完整的匹配结果已保存至文件: '{output_file}'")
    except Exception as e:
        print(f"\n保存文件时出错: {e}")

if __name__ == '__main__':
    synchronize_robot_data()