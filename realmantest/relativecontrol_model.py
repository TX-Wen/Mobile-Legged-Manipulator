import time
import numpy as np
from Robotic_Arm.rm_robot_interface import *
import threading
import keyboard  # 确保已安装此库

# (matrix_to_pose_list 和 get_mock_trajectory_plan 函数与之前相同，此处省略)

import numpy as np

# --- 用于急停的全局标志 ---
stop_flag = threading.Event()

def emergency_stop_listener(arm_instance):
    """
    该函数在一个独立的线程中运行，监听键盘's'键的按下事件。
    """
    print("\n按 'S' 键可随时触发紧急停止。")
    keyboard.wait('s')
    print("\n检测到 'S' 键按下！正在发送紧急停止指令...")
    stop_flag.set()  # 设置全局标志，让主循环知道需要停止
    
    # 调用手册中提供的急停函数 
    ret = arm_instance.rm_set_arm_stop()
    if ret == 0:
        print("紧急停止指令已成功发送。")
    else:
        print(f"发送紧急停止指令时出错，错误码: {ret}")


def matrix_to_pose_list_euler(matrix):
    """
    将一个4x4的NumPy变换矩阵转换为6元素的位姿列表 [x, y, z, roll, pitch, yaw]。
    
    [cite_start]单位与API要求一致：位置为米，姿态为弧度 [cite: 45, 80, 151]。

    Args:
        matrix (np.ndarray): 一个4x4的变换矩阵。

    Returns:
        list: 一个6元素的列表 [x, y, z, roll, pitch, yaw]。
    """
    # 1. 提取位置 (x, y, z)
    # 位置信息在矩阵的最后一列的前三个元素
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # 2. 提取旋转矩阵并计算欧拉角 (roll, pitch, yaw)
    R = matrix[:3, :3]
    
    # 计算 Pitch (绕Y轴的旋转)
    # sin(pitch) = -R[2, 0]
    sin_p = -R[2, 0]
    pitch = np.arcsin(sin_p)

    # 检查万向节死锁 (Gimbal Lock)
    # 当 pitch 接近 +/- 90度时，cos(pitch) 接近 0，会导致计算 Yaw 和 Roll 的公式不稳定
    if np.isclose(np.cos(pitch), 0.0):
        # 发生万向节死锁
        # 此时，我们通常约定 roll 为 0，然后计算 yaw
        roll = 0.0
        # yaw = atan2(R[0, 1], R[1, 1])
        yaw = np.arctan2(R[0, 1], R[1, 1])
    else:
        # 一般情况
        # tan(yaw) = R[1, 0] / R[0, 0]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        # tan(roll) = R[2, 1] / R[2, 2]
        roll = np.arctan2(R[2, 1], R[2, 2])
        
    return [x, y, z, roll, pitch, yaw]

def matrix_to_pose_list(matrix):
    """
    将一个4x4的NumPy变换矩阵转换为7元素的位姿列表
    （3个位置元素，4个四元数元素）。
    """
    position = matrix[0:3, 3]
    rotation_matrix = matrix[0:3, 0:3]
    qw = 0.5 * np.sqrt(max(0, 1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]))
    epsilon = 1e-6
    if abs(qw) < epsilon:
        qx, qy, qz = 0, 0, 0
    else:
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)
    quaternion = [qx, qy, qz, qw]
    return list(position) + list(quaternion)

def pose_list_to_matrix(pose_list):
    """
    辅助函数：根据列表长度判断并转换位姿列表为4x4矩阵。
    - 如果列表长度为7，则认为是 [x,y,z,qx,qy,qz,qw] (四元数)
    - 如果列表长度为6，则认为是 [x,y,z,roll,pitch,yaw] (欧拉角)
    """
    if len(pose_list) == 6:
        # --- 处理欧拉角 ---
        x, y, z, roll, pitch, yaw = pose_list
        
        # 计算 roll (绕X轴), pitch (绕Y轴), yaw (绕Z轴) 的 sin 和 cos 值
        c_y = np.cos(yaw)
        s_y = np.sin(yaw)
        c_p = np.cos(pitch)
        s_p = np.sin(pitch)
        c_r = np.cos(roll)
        s_r = np.sin(roll)
        
        # 构建旋转矩阵 (通常为 ZYX 顺序)
        # 注意: 欧拉角的旋转顺序可能因机器人品牌而异，ZYX (yaw, pitch, roll) 是一个常见的标准。
        # 如果发现姿态不匹配，可能需要调整这里的乘法顺序。
        R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
        R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
        R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
        
        rotation_matrix = R_z @ R_y @ R_x
        
        # 构建4x4变换矩阵
        matrix = np.identity(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = [x, y, z]
        return matrix

    elif len(pose_list) == 7:
        # --- 处理四元数 (代码与之前相同) ---
        p = pose_list
        position = np.array([p[0], p[1], p[2]])
        qx, qy, qz, qw = p[3], p[4], p[5], p[6]
        rotation_matrix = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        matrix = np.identity(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = position
        return matrix
    else:
        raise ValueError("pose_list 的长度必须是 6 (欧拉角) 或 7 (四元数)")
    
def get_mock_trajectory_plan():
    trajectory_plan = []
    num_steps = 2
    for i in range(num_steps):
        if i < 1:
            #注意：！！xyz值单位为米！！不要设太大！！
            step_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.05], [0, 0, 0, 1]])#向z轴负方向移动5cm
        else:
            angle = np.deg2rad(10)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            step_transform = np.array([[cos_a, -sin_a, 0, 0], [sin_a,  cos_a, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])#绕z轴
            #step_transform = np.array([[1, 0, 0, 0], [0,  cos_a,-sin_a, 0], [sin_a,  cos_a, 1, 0], [0, 0, 0, 1]])#绕x轴
            #step_transform = np.array([[cos_a, 0, sin_a, 0], [0, 1, 0, 0], [-sin_a, 0, cos_a, 0], [0, 0, 0, 1]])#绕y轴
        trajectory_plan.append(step_transform)
    print(f"生成了包含 {len(trajectory_plan)} 个步骤的新轨迹计划。")
    return trajectory_plan


# --- 主控制脚本 ---
def main():
    print("正在初始化机械臂...")
    # 实例化RoboticArm类，使用支持异步指令的模式 [cite: 18]
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    inverse_camera2eef_matrix = np.array([
    [-0.15441093, -0.36734396, -0.91717811, -54.13019075*10e-3],
    [0.8574527,    0.41136713, -0.30911477, -13.17631122*10e-3],
    [0.49084837,   -0.83416755,  0.25146049, -21.25658945*10e-3],
    [0.,           0.,           0.,          1.        ]
    ])#单位mm 注意转换
    camera2eef_matrix = np.array([[ -0.15441093,   0.85745271,   0.49084837,  13.37353295*10e-3],
       [ -0.36734396,   0.41136713,  -0.83416755, -32.19565444*10e-3],
       [ -0.91717811,  -0.30911477,   0.25146049, -48.37482606*10e-3],
       [  0.        ,   0.        ,   0.        ,   1.        ]
    ])#单位mm 注意转换

    # 创建机械臂连接 [cite: 21]
    handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
    print(f"成功连接到机械臂，句柄ID: {handle.id}")
    
    # 在一个新线程中启动键盘监听
    stop_listener_thread = threading.Thread(target=emergency_stop_listener, args=(arm,))
    stop_listener_thread.daemon = True  # 设置为守护线程，这样主程序退出时它也会退出
    stop_listener_thread.start()

    try:
        print("控制循环已启动... 按 'S' 触发急停, 按 Ctrl+C 退出。")
        while not stop_flag.is_set():
        # (您的获取初始位姿代码...)
            ret_val = arm.rm_get_current_arm_state()
            initial_pose_list = ret_val[1]['pose'] 
            current_pose_matrix = pose_list_to_matrix(initial_pose_list)
            print("已获取初始位姿。")
            
            print("控制循环已启动... 按 Ctrl+C 退出。")

            trajectory_to_execute = get_mock_trajectory_plan()
            print("开始执行轨迹计划...")
            
            temp_pose_matrix = current_pose_matrix.copy()
            for step_index, relative_transform_matrix in enumerate(trajectory_to_execute):
                # 在每次移动前检查急停标志
                if stop_flag.is_set():
                    print("急停已触发，正在终止轨迹执行。")
                    break

                # (您的位姿计算代码...)
                target_pose_matrix = temp_pose_matrix @ relative_transform_matrix

                #带手眼坐标转换
                #target_pose_matrix = temp_pose_matrix @ camera2eef_matrix @ relative_transform_matrix @ inverse_camera2eef_matrix

                #temp_pose_matrix = target_pose_matrix #相对第一步的绝对变化时，无需更新
                target_pose_list = matrix_to_pose_list_euler(target_pose_matrix)
                
                print(f"正在执行第 {step_index + 1} 步...")
                
                # 发送移动指令。如果此时在另一线程中调用了rm_set_arm_stop()，
                # 这个运动会被中断。
                ret = arm.rm_movel(target_pose_list, 5, 0, 0, 1)
                print(f"移动指令返回值: {ret}")

                # 运动指令返回后，再次检查标志位。如果运动被中断，也应跳出循环。
                if stop_flag.is_set():
                    break
            
            # 再次检查标志位，以防 for 循环是被急停命令中断的
            if stop_flag.is_set():
                break # 退出主 while 循环

            if not stop_flag.is_set():
                print("轨迹计划执行完毕。")

            # 新增：在开始下一个循环前暂停1秒(确保安全后可注释)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n用户通过 Ctrl+C 停止了程序。")
        
    except Exception as e:
        print(f"发生未预料的错误: {e}")
        
    finally:
        print("正在断开与机械臂的连接。")
        arm.rm_delete_robot_arm()
        print("程序结束。")

if __name__ == '__main__':
    main()