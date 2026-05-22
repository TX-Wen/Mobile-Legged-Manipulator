import sys
import threading
import time
import os
import pathlib
from collections import deque
import json
from copy import deepcopy
import cv2
import dill
import hydra
import keyboard
import numpy as np
import torch
from omegaconf import OmegaConf
from pyorbbecsdk import *
from scipy.spatial.transform import Rotation

from Robotic_Arm.rm_robot_interface import *
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')


# ================================== 配置区 ==================================
# --- 模型与设备配置 ---
CHECKPOINT_PATH = "/home/baishan/diffusion_policy_real/data/epoch=0050.ckpt"
DEVICE = 'cuda:0'
N_EXEC_STEPS = 4

# --- 机械臂配置 ---
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080

# --- 相机配置 ---
NATIVE_CAM_WIDTH = 1280
NATIVE_CAM_HEIGHT = 720
MODEL_IMG_WIDTH = 640
MODEL_IMG_HEIGHT = 360
N_OBS_STEPS = 2

# --- 手眼标定矩阵 ---
T_camera2eef = np.array([[-1.24506649e-02, -9.99799558e-01,  1.56788035e-02,
         7.05597162e+01*1e-3],
       [ 9.97103375e-01, -1.35906345e-02, -7.48341824e-02,
         1.36206029e+01*1e-3],
       [ 7.50322674e-02,  1.47016525e-02,  9.97072726e-01,
        -5.15583194e+01*1e-3],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
inv_T_camera2eef = np.linalg.inv(T_camera2eef)
# --- 【关键新增】手动反归一化参数 ---
# 使用您在数据创建脚本中计算出的原始全局 min 和 max
ORIGINAL_ACTION_POS_MIN = np.array([-7.383911, -9.397736, -1.0109711])
ORIGINAL_ACTION_POS_MAX = np.array([7.6414795, 8.594421, 14.518967])


# ============================ 辅助函数 ============================

def manual_unnormalize_action_pos(normalized_pos_action, min_vals, max_vals):
    """
    手动反归一化动作的位移部分。
    公式: 从 [-1, 1] 映射回 [min, max]
    """
    action_range = max_vals - min_vals
    original_action = (normalized_pos_action + 1.0) / 2.0 * action_range + min_vals
    return original_action

# ... (其他辅助函数 action_6d_to_transform_matrix, matrix_to_pose_list_euler 等保持不变) ...
def action_6d_to_transform_matrix(action_6d: np.ndarray) -> np.ndarray:
    if action_6d.shape != (6,):
        raise ValueError(f"输入的6D动作向量形状必须是(6,), 但得到的是{action_6d.shape}")
    translation = action_6d[:3]
    rotation_vector = action_6d[3:]
    try:
        rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    except Exception:
        print("警告：旋转向量无效，使用单位矩阵作为旋转部分。")
        rotation_matrix = np.eye(3)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def matrix_to_pose_list_euler(matrix):
    x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
    R = matrix[:3, :3]
    sin_p = -R[2, 0]
    pitch = np.arcsin(sin_p)
    if np.isclose(np.cos(pitch), 0.0):
        roll = 0.0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    return [x, y, z, roll, pitch, yaw]

def pose_list_to_matrix(pose_list):
    if len(pose_list) != 6:
         raise ValueError("此部署脚本仅支持长度为6的欧拉角pose_list")
    x, y, z, roll, pitch, yaw = pose_list
    c_y, s_y = np.cos(yaw), np.sin(yaw); c_p, s_p = np.cos(pitch), np.sin(pitch); c_r, s_r = np.cos(roll), np.sin(roll)
    R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]]); R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]]); R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    matrix = np.identity(4); matrix[:3, :3] = R_z @ R_y @ R_x; matrix[:3, 3] = [x, y, z]
    return matrix

def denormalize_action(norm_action, global_min, global_range):
    # norm_action [-1, 1] -> 原始动作值
    return (norm_action + 1.0) / 2.0 * global_range + global_min

# ============================== 核心功能封装 (有修改) ==============================

# ... (CameraManager 类保持不变) ...
class CameraManager:
    """管理相机硬件，保持流打开，并按需提供处理过的帧。"""
    def __init__(self, native_width, native_height, model_width, model_height):
        print(f"正在初始化相机，请求原始分辨率: {native_width}x{native_height}...")
        self.model_width = model_width
        self.model_height = model_height
        self.pipeline = Pipeline()
        config = Config()
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(native_width, native_height, OBFormat.BGR, 30)
            if color_profile is None:
                 print(f"警告：找不到 {native_width}x{native_height} 分辨率，将尝试默认配置。")
                 config.enable_stream(OBSensorType.COLOR_SENSOR)
            else:
                 config.enable_stream(color_profile)
            self.pipeline.start(config)
            print("相机初始化成功，视频流已启动。")
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise

    def get_latest_frame(self):
        frames = self.pipeline.wait_for_frames(100)
        if not frames: return None
        color_frame = frames.get_color_frame()
        if not color_frame: return None
        return np.asanyarray(color_frame.get_data())

    def preprocess_for_model(self, bgr_image, device):
        resized_image = cv2.resize(bgr_image, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_image.astype(np.float32) / 255.0
        chw_image = np.transpose(rgb_float, (2, 0, 1))
        return torch.from_numpy(chw_image).to(device)

    def stop(self):
        print("正在关闭相机...")
        self.pipeline.stop()

class PolicyModel:
    """封装模型的加载、配置和推理。"""
    def __init__(self, checkpoint_path, device):
        print(f"正在从 {checkpoint_path} 加载模型...")
        self.device = torch.device(device)
        
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model
        #data_cfg = deepcopy(self.cfg)
        #data_cfg.task.dataset.dataset_path = "/home/baishan/diffusion_policy_real/data/my_robot_dataset_normalized.hdf5"
        #dataset = hydra.utils.instantiate(data_cfg.task.dataset)
        #self.normalizer = dataset.get_normalizer()
        
        #self.policy.set_normalizer(self.normalizer)
        self.policy.to(self.device)
        self.policy.eval()
        #data_root = "/home/yueyaohua/machine_learning/AI/diffusion/diffusion_policy-main/data/data2"
        stats_path = ("/home/baishan/diffusion_policy_real/result/global_norm_stats.json")            #需要给出具体地址
        with open(stats_path, "r") as f:
            norm_stats = json.load(f)
        self.global_min = np.array(norm_stats["min"], dtype=np.float32)
        self.global_max = np.array(norm_stats["max"], dtype=np.float32)
        self.global_range = self.global_max - self.global_min
        self.global_range[self.global_range == 0] = 1.0
        print("模型加载并配置完成。")

    @torch.no_grad()
    def predict_action(self, obs_history: deque):
        # --- 【关键修改】让函数返回一个动作序列 ---
        obs_tensor = torch.stack(list(obs_history), dim=0).unsqueeze(0)
        obs_dict = {'images': obs_tensor}
        
        #result, flag = self.policy.predict_action_flag(obs_dict)
        result = self.policy.predict_action(obs_dict)
        # result['action'] 的形状是 (1, n_action_steps, action_dim)
        action = result['action']
        #flag = result['flag']
        #flag = flag[0]
        action_flat = action.reshape(-1, 6).cpu().detach().numpy()
        action_denorm = denormalize_action(action_flat, self.global_min, self.global_range)
        #action_denorm = action_denorm[0]
        # 对序列中的每一个动作都进行手动反归一化
        #print(action_denorm)
        return action_denorm
        #return action_denorm, flag

# ... (RobotController 类和 main 函数保持不变) ...
class RobotController:
    """管理机械臂连接、运动控制和紧急停止。"""
    def __init__(self, ip, port):
        print(f"正在连接机械臂 at {ip}:{port}...")
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        if self.handle.id == -1: raise ConnectionError(f"连接机械臂失败！")
        print(f"成功连接到机械臂，句柄ID: {self.handle.id}")
        self.stop_flag = threading.Event()

    def emergency_stop_listener(self):
        print("\n按 'S' 键可随时触发紧急停止。")
        keyboard.wait('s')
        print("\n检测到 'S' 键按下！正在发送紧急停止指令...")
        self.stop_flag.set()
        ret = self.arm.rm_set_arm_stop()
        if ret == 0: print("紧急停止指令已成功发送。")
        else: print(f"发送紧急停止指令时出错，错误码: {ret}")
            
    def start_estop_listener(self):
        thread = threading.Thread(target=self.emergency_stop_listener)
        thread.daemon = True
        thread.start()

    def get_current_pose_matrix(self):
        ret_val = self.arm.rm_get_current_arm_state()
        if ret_val[0] != 0:
             print("获取机械臂当前状态失败！")
             return None
        return pose_list_to_matrix(ret_val[1]['pose'])

    # execute_action_wrong这是错误的代码
    def execute_action_wrong(self, current_eef_pose_matrix, action_6d):
        # --- 函数开头的安全检查保持不变 ---
        if self.stop_flag.is_set():
            print("急停已触发，无法执行动作。"); return None

        # 将模型预测的6D动作向量转换为4x4齐次变换矩阵
        # 这是相机坐标系下的相对运动: T_cam_current_to_cam_next
        relative_transform = action_6d_to_transform_matrix(action_6d)
        
        # 步骤 1 & 2: 计算相机运动后的目标绝对位姿 (在基座坐标系下)
        # T_base_to_cam_target = (T_base_to_eef_current @ T_eef_to_camera) @ T_cam_current_to_cam_next
        target_camera_pose_in_base = (
            current_eef_pose_matrix @
            T_eef_camera @
            relative_transform
        )

        # 步骤 3: 根据相机的目标位姿，反推出末端执行器需要达到的目标位姿
        # T_base_to_eef_target = T_base_to_cam_target @ T_camera_to_eef
        target_eef_pose_matrix = target_camera_pose_in_base @ T_camera_eef
        
        # --- 函数剩余部分（打印信息、发送指令等）保持不变 ---
        target_pose_list = matrix_to_pose_list_euler(target_eef_pose_matrix)
        print("当前位姿", matrix_to_pose_list_euler(current_eef_pose_matrix))
        print("预测动作", action_6d)
        print(f"正在执行步骤，目标: {[f'{x:.4f}' for x in target_pose_list]}")
        
        velocity_percentage = 15; blend_radius = 0.0
        ret = self.arm.rm_movel(target_pose_list, 3, 0, 0, 1)
        
        if ret != 0: 
            print(f"移动指令执行失败，错误码: {ret}")
            print(f"错误码: {ret}")
            print("当前位姿", matrix_to_pose_list_euler(current_eef_pose_matrix))
            return None # 如果失败，返回 None
        
        return target_eef_pose_matrix

    def execute_action(self, current_eef_pose_matrix, action_6d):
        # --- 【关键修改】此函数现在返回计算出的目标位姿 ---
        if self.stop_flag.is_set():
            print("急停已触发，无法执行动作。"); return None

        action_6d[:3] = [x * 1e-3 for x in action_6d[:3]]
        relative_transform = action_6d_to_transform_matrix(action_6d)
        target_eef_pose_matrix = (
            current_eef_pose_matrix @
            relative_transform
        )
        """
        target_eef_pose_matrix = (
            current_eef_pose_matrix @
            T_camera2eef @
            relative_transform @
            inv_T_camera2eef
        )
        
        target_eef_pose_matrix = (
            inv_T_camera2eef@
            current_eef_pose_matrix @
            T_camera2eef @
            relative_transform 
        )
        """
        target_pose_list = matrix_to_pose_list_euler(target_eef_pose_matrix)
        print("当前位姿", matrix_to_pose_list_euler(current_eef_pose_matrix))
        print("预测动作",action_6d)
        print(f"正在执行步骤，目标: {[f'{x:.4f}' for x in target_pose_list]}")
        print("目标位姿", target_pose_list)
        #target_pose_list[:3] = [x * 1e-3 for x in target_pose_list[:3]]
        #target_pose_list = [round(value, 5) for value in target_pose_list]
        #target_pose_list[:3] = target_pose_list[:3] * 1e-3 # mm to m
        velocity_percentage = 15; blend_radius = 0.0
        #ret = self.arm.rm_movel(target_pose_list, 3, 0, 0, 1)
        ret = self.arm.rm_movej_p(target_pose_list, 3, 0, 0, 1)
        #print(arm.rm_get_arm_current_trajectory())
        #ret = self.arm.rm_movep_canfd(target_pose_list, True, 1, 60)
        
        if ret != 0: 
            print(f"移动指令执行失败，错误码: {ret}")
            print(f"错误码: {ret}")
            print("当前位姿", matrix_to_pose_list_euler(current_eef_pose_matrix))
            return None # 如果失败，返回 None
        
        # 返回计算出的目标位姿，用于下一次迭代
        return target_eef_pose_matrix

    def disconnect(self):
        print("正在断开与机械臂的连接。"); self.arm.rm_delete_robot_arm()

def main():
    
    checkpoint_path = sys.argv[1]
    model = PolicyModel(checkpoint_path, device=DEVICE)
    robot = RobotController(ip=ROBOT_IP, port=ROBOT_PORT)
    robot.start_estop_listener()
    camera = CameraManager(
        native_width=NATIVE_CAM_WIDTH, native_height=NATIVE_CAM_HEIGHT,
        model_width=MODEL_IMG_WIDTH, model_height=MODEL_IMG_HEIGHT
    )

    try:
        print("\n相机正在预热，请等待图像稳定...")
        for _ in range(30):  # 约1秒钟
            camera.get_latest_frame(); time.sleep(0.03)

        obs_history = deque(maxlen=N_OBS_STEPS)

        print("正在初始化观测历史...")
        while len(obs_history) < N_OBS_STEPS:
            obs_bgr = camera.get_latest_frame()
            if obs_bgr is not None:
                obs_tensor = camera.preprocess_for_model(obs_bgr, model.device)
                obs_history.append(obs_tensor)
                print(f"已捕获 {len(obs_history)}/{N_OBS_STEPS} 帧初始图像...")
                time.sleep(0.1)
        
        print("="*50)
        input("初始化完成，按 Enter 键开始实时控制循环...")
        
        print("\n开始实时控制循环。按 'S' 急停，按 Ctrl+C 退出程序。")
        # --- 【关键修改】主循环逻辑变更 ---
        while not robot.stop_flag.is_set():
            # 1. (外部循环) 观察并进行一次模型推理，获得一个动作计划
            print("-" * 20)
            print("获取新观测，正在进行模型推理以生成动作计划...")
            
            # a. 更新观测历史
            obs_bgr = camera.get_latest_frame()
            if obs_bgr is None: print("警告：无法获取当前帧。"); time.sleep(0.1); continue
            #cv2.imshow("Live View", obs_bgr);
            #if cv2.waitKey(1)&0xFF==ord('q'): break
            obs_history.append(camera.preprocess_for_model(obs_bgr, model.device))

            # b. 获得一个包含N步动作的计划
            action_plan = model.predict_action(obs_history)
            #action_plan, flag = model.predict_action(obs_history)
            '''
            if flag == 1:
                camera.stop()
                robot.disconnect()
                cv2.destroyAllWindows()
                print("到达目标位置，程序已安全退出。")
            '''
            # c. 获取计划开始时的机器人起始位姿
            # 这个位姿将作为接下来N步开环控制的起点
            start_pose_matrix = robot.get_current_pose_matrix()
            if start_pose_matrix is None: print("无法获取机械臂位姿。"); time.sleep(0.1); continue

            # 2. (内部循环) 开环执行这个动作计划中的前 N_EXEC_STEPS 步
            print(f"推理完成，开始执行计划中的前 {N_EXEC_STEPS} 步动作...")
            next_pose_matrix = start_pose_matrix.copy() # 用于迭代更新位姿
            
            for i in range(N_EXEC_STEPS):
                if robot.stop_flag.is_set(): break # 每一步都检查急停
                
                action_step = action_plan[i]
                print(f"\n执行第 {i+1}/{N_EXEC_STEPS} 步...")
                
                # 核心逻辑：下一个动作的目标位姿是基于上一个动作的目标位姿计算的
                # 而不是每次都重新获取机械臂的真实位置
                returned_pose = robot.execute_action(next_pose_matrix, action_step)
                
                if returned_pose is None:
                    print("动作执行失败，中断当前计划。")
                    break
                
                #next_pose_matrix = returned_pose # 更新下一次迭代的起始位姿
                time.sleep(0.001)

            print(f"已完成 {N_EXEC_STEPS} 步计划，将重新进行观察和推理。")

    except KeyboardInterrupt:
        print("\n用户通过 Ctrl+C 请求停止程序。")
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.stop()
        robot.disconnect()
        cv2.destroyAllWindows()
        print("程序已安全退出。")

if __name__ == '__main__':
    main()