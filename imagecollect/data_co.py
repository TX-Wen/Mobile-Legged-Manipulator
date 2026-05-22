import pandas as pd
import re
from collections import defaultdict
import glob

# --- 用户配置 ---
# 您上传文件所在的基础目录名
# 如果您的文件夹结构不是 '离散标定数据/组编号/文件名', 请修改此处的 '离散标定数据'
BASE_FOLDER_NAME = '离散标定数据'

# --- 脚本开始 ---

def consolidate_calibration_data(file_list):
    """
    从文件路径列表中读取、处理并合并标定数据。

    Args:
        file_list (list): 所有上传文件的路径列表。

    Returns:
        pandas.DataFrame: 包含所有合并后数据的数据框。
    """
    # 使用 defaultdict 创建一个字典，用于按组编号存放文件路径
    # 结构: {'组编号': {'ndi': 'ndi文件路径', 'sja': 'sja文件路径'}}
    grouped_files = defaultdict(lambda: {'ndi': None, 'sja': None})

    # 使用正则表达式从文件路径中提取组编号 (数字)
    # 例如, 从 '离散标定数据/9/...' 中提取 '9'
    pattern = re.compile(rf"{re.escape(BASE_FOLDER_NAME)}/(\d+)/")

    for path in file_list:
        match = pattern.search(path)
        if match:
            group_id = match.group(1)
            # 判断文件类型并存入字典
            if 'NDI_tracking' in path and path.endswith('.csv'):
                grouped_files[group_id]['ndi'] = path
            elif 'single_joint_angle' in path and path.endswith('.csv'):
                grouped_files[group_id]['sja'] = path

    # 用于存放每个组处理后的数据
    all_data_rows = []

    # 按数字大小对组编号进行排序，确保输出文件顺序正确
    sorted_group_ids = sorted(grouped_files.keys(), key=int)

    print(f"找到了 {len(sorted_group_ids)} 个数据组，正在处理...")

    for group_id in sorted_group_ids:
        files = grouped_files[group_id]
        ndi_file_path = files['ndi']
        sja_file_path = files['sja']

        # 确保两个必要的CSV文件都已找到
        if ndi_file_path and sja_file_path:
            try:
                # 读取CSV文件到 pandas DataFrame
                df_ndi = pd.read_csv(ndi_file_path)
                df_sja = pd.read_csv(sja_file_path)

                # --- 修改部分 ---
                # 提取每个DataFrame中数值列的最后一行数据
                # 这会将多行数据压缩成一个Series (单行)
                ndi_last_row = df_ndi.select_dtypes(include='number').iloc[-1]
                sja_last_row = df_sja.select_dtypes(include='number').iloc[-1]

                # 为列名添加前缀以区分来源 ('ndi_' 或 'sja_')
                # .str.strip() 用于去除列名中可能存在的多余空格
                ndi_last_row.index = 'ndi_' + ndi_last_row.index.str.strip()
                sja_last_row.index = 'sja_' + sja_last_row.index.str.strip()

                # 将两个Series合并成一个
                combined_series = pd.concat([ndi_last_row, sja_last_row])
                # --- 修改结束 ---

                # 添加组编号信息
                combined_series['group_id'] = int(group_id)

                # 将处理好的单行数据添加到列表中
                all_data_rows.append(combined_series)
            except Exception as e:
                print(f"处理组 {group_id} 时出错: {e}")
        else:
            print(f"警告: 组 {group_id} 缺少 'NDI_tracking' 或 'single_joint_angle' CSV 文件。")

    if not all_data_rows:
        print("没有成功处理任何数据。")
        return None

    # 将列表中的所有Series转换成一个完整的DataFrame
    final_df = pd.DataFrame(all_data_rows)

    # 调整列顺序，将 'group_id' 放到第一列
    if 'group_id' in final_df.columns:
        cols = final_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('group_id')))
        final_df = final_df[cols]

    return final_df

# --- 主程序 ---
if __name__ == '__main__':
    # 在实际环境中，我们会有一个包含所有上传文件路径的列表
    # 这里我们使用 glob 动态查找所有符合条件的文件路径来模拟这个列表
    # 这使得脚本在本地运行时也能正常工作
    all_files = glob.glob(f'{BASE_FOLDER_NAME}/*/*.csv')
    
    if not all_files:
        print("错误: 在指定的基础文件夹中没有找到任何CSV文件。")
        print("请确认您的文件是否在名为 '离散标定数据' 的文件夹下，并且结构正确。")
    else:
        # 调用主函数处理数据
        consolidated_df = consolidate_calibration_data(all_files)

        if consolidated_df is not None:
            # 定义输出文件名
            output_filename = 'consolidated_data_last_row.csv'
            
            # 将最终的DataFrame保存为CSV文件，不包含索引列
            consolidated_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            
            print("\n处理完成！")
            print(f"所有数据已成功合并到 '{output_filename}' 文件中。")
            print("\n数据预览 (前5行):")
            print(consolidated_df.head())
