# 导入 pandas 库，它主要用于数据处理和分析
import pandas as pd

def convert_datetime_to_unix(
    input_filename: str,
    output_filename: str,
    datetime_column: str,
    timezone: str = 'Asia/Shanghai'
):
    """
    读取一个CSV文件，将一个日期时间列转换为带有正确时区的Unix时间戳，
    并将结果保存到一个新的CSV文件中。

    参数:
        input_filename (str): 需要读取的CSV文件名。
        output_filename (str): 需要保存的新CSV文件名。
        datetime_column (str): 包含日期时间字符串的列名。
        timezone (str): 原始日期时间数据所在的时区 (例如, 'Asia/Shanghai' 代表东八区)。
    """
    try:
        # 步骤 1: 读取原始的CSV文件
        df = pd.read_csv(input_filename)
        print(f"成功加载文件: '{input_filename}'")

        # 步骤 2: 将指定的列转换为 datetime 对象
        # 这是进行时区操作前的必要步骤
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # 步骤 3: 将datetime对象本地化到正确的时区，然后转换为Unix时间戳
        # .dt.tz_localize(timezone) 是告诉pandas原始数据的时区
        # .astype(int) / 10**9 将其转换为高精度的、以秒为单位的Unix时间戳
        # 这是获得正确数值最关键的一步
        new_timestamp_col = "Corrected_Unix_Timestamp"  # 新列的名称
        df[new_timestamp_col] = df[datetime_column].dt.tz_localize(timezone).astype(int) / 10**9
        print(f"已创建新的时间戳列: '{new_timestamp_col}'")

        # 步骤 4: 将更新后的数据保存到一个新的CSV文件
        # `float_format='%.6f'` 参数确保时间戳的完整精度被保存下来
        df.to_csv(output_filename, index=False, float_format='%.6f')
        print(f"已成功将新数据保存至: '{output_filename}'")

        # (可选) 显示结果的前几行作为预览
        print("\n新文件内容预览 (前5行):")
        print(df.head())

    except FileNotFoundError:
        print(f"错误: 文件 '{input_filename}' 未找到。")
    except KeyError:
        print(f"错误: 在文件中未找到名为 '{datetime_column}' 的列。")
    except Exception as e:
        print(f"发生未知错误: {e}")


# --- 如何使用 ---
# 1. 请确保您的Python脚本文件和您的数据文件在同一个文件夹内。
# 2. 修改下面的变量，让它们与您的文件名和列名相匹配。

# 你的原始数据文件名
input_file = "joint_angle_recordstest1.csv" 

# 你希望生成的新文件名
output_file = "joint_angle_records_corrected_cn.csv"

# 包含日期时间值的列的名称 (例如, "2025-08-14 18:15:25.395738")
datetime_column_name = "Timestamp"

# 你的数据采集时所在的地理时区。对于中国标准时间，这里应为 'Asia/Shanghai'。
# 如果您在其他时区，可以在此链接查找您的时区名称: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
user_timezone = "Asia/Shanghai"

# 使用您设置的参数来运行上面的函数
convert_datetime_to_unix(input_file, output_file, datetime_column_name, user_timezone)