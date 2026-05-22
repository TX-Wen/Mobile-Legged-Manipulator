import sys
import threading
import time
import os
import pathlib
from collections import deque

import cv2
import dill
import hydra
import keyboard
import numpy as np
import torch
from omegaconf import OmegaConf
from pyorbbecsdk import *
from scipy.spatial.transform import Rotation

# 假设您的机械臂接口和模型代码在PYTHONPATH中，或者与此脚本在同一目录下
from Robotic_Arm.rm_robot_interface import *
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

# 为 stdout 和 stderr 使用行缓冲，确保日志能及时输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')


# ================================== 配置区 ==================================
# --- 模型与设备配置 ---
CHECKPOINT_PATH = "data/epoch=0050.ckpt"
DEVICE = 'cuda:0'

# --- 机械臂配置 ---
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080

# --- 相机配置 ---
# 【关键修改】尺寸必须与模型训练时完全一致
NATIVE_CAM_WIDTH = 1280
NATIVE_CAM_HEIGHT = 720
# 模型训练时使用的图像尺寸 (来自 YAML: shape_meta.obs.robot0_eye_in_hand_image.shape)
MODEL_IMG_WIDTH = 640
MODEL_IMG_HEIGHT = 360
# 模型期望的观测历史步数 (来自 YAML: n_obs_steps)
N_OBS_STEPS = 2


# --- 手眼标定矩阵 (请务必确认其准确性！) ---
# T_eef_camera: 从末端执行器坐标系到相机坐标系的变换
# 【注意！】请确认平移向量的单位！机械臂通常使用米，如果标定结果是毫米，需要转换。
# 假设以下矩阵中的平移部分（最后一列）单位是米。如果不是，请手动转换！
T_eef_camera = np.array([
    [-0.377017613, -0.270410618, -0.885853157, -0.044825905],
    [-0.033632480, -0.951804295,  0.304856423,  0.000234192],
    [-0.925595254,  0.144729680,  0.349752403,  0.043274332],
    [0.0,           0.0,           0.0,          1.0]
])
T_camera_eef = np.array([[-3.77017613e-01, -3.36324805e-02, -9.25595255e-01, 2.31622378e-2],
       [-2.70410618e-01, -9.51804296e-01,  1.44729680e-01, -1.81615755e-2],
       [-8.85853158e-01,  3.04856424e-01,  3.49752404e-01, -5.49158669e-2],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000]])#单位mm 注意转换
# T_camera_eef: T_eef_camera 的逆矩阵
# T_camera_eef = np.linalg.inv(T_eef_camera)


# ============================ 辅助函数 (基本无变化) ============================
def action_6d_to_transform_matrix(action_6d: np.ndarray) -> np.ndarray:
    if action_6d.shape != (6,):
        raise ValueError(f"输入的6D动作向量形状必须是(6,), 但得到的是{action_6d.shape}")
    translation = action_6d[:3]
    rotation_vector = action_6d[3:]
    try:
        rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    except Exception:
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

# ============================== 核心功能封装 (有修改) ==============================

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
        # 1. 将图像从原始尺寸强制缩放到模型输入尺寸
        resized_image = cv2.resize(bgr_image, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)
        # 2. BGR -> RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        # 3. 归一化到 [0, 1]
        rgb_float = rgb_image.astype(np.float32) / 255.0
        # 4. HWC -> CHW
        chw_image = np.transpose(rgb_float, (2, 0, 1))
        # 5. 转换为Tensor
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
            
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        self.normalizer = dataset.get_normalizer()
        
        self.policy.set_normalizer(self.normalizer)
        self.policy.to(self.device)
        self.policy.eval()
        print("模型加载并配置完成。")

    @torch.no_grad()
    def predict_action(self, obs_history: deque):
        """
        【关键修改】输入一个包含N_OBS_STEPS帧图像的队列，输出6D动作向量。
        """
        # 1. 将队列中的图像张量堆叠成一个批次
        # obs_history中的每个元素都是 (C,H,W)
        # torch.stack后变成 (N_OBS_STEPS, C, H, W)
        # .unsqueeze(0)后变成 (1, N_OBS_STEPS, C, H, W) 以符合模型B,T,C,H,W的输入格式
        obs_tensor = torch.stack(list(obs_history), dim=0).unsqueeze(0)

        # 2. 构建模型期望的输入字典 (key必须与训练时一致)
        obs_dict = {
            'robot0_eye_in_hand_image': obs_tensor
        }
        
        # 3. 模型推理
        result = self.policy.predict_action(obs_dict)
        
        # 4. 提取预测的动作
        # result['action'] 的形状是 (1, n_action_steps, action_dim)
        # 我们只取这批动作中的第一个动作来执行
        action = result['action'][0, 0].cpu().numpy()
        return action

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

    def execute_action(self, current_eef_pose_matrix, action_6d):
        if self.stop_flag.is_set():
            print("急停已触发，无法执行动作。"); return
        
        relative_transform_in_camera_frame = action_6d_to_transform_matrix(action_6d)
        
        # T_base_eef_new = T_base_eef_old @ T_eef_camera @ T_camera_action @ T_camera_eef
        target_eef_pose_matrix = (
            current_eef_pose_matrix @
            T_eef_camera @
            relative_transform_in_camera_frame @
            T_camera_eef
        )
        
        target_pose_list = matrix_to_pose_list_euler(target_eef_pose_matrix)
        print(f"正在移动到目标位姿: {[f'{x:.4f}' for x in target_pose_list]}")
        ret = self.arm.rm_movel(target_pose_list, 0.15, 0, 0, 1) # 降低速度确保安全
        if ret != 0: print(f"移动指令执行失败，错误码: {ret}")
        return ret

    def disconnect(self):
        print("正在断开与机械臂的连接。"); self.arm.rm_delete_robot_arm()

# ================================== 主程序 (有修改) ==================================

def main():
    # --- 1. 初始化 ---
    camera = CameraManager(
        native_width=NATIVE_CAM_WIDTH, native_height=NATIVE_CAM_HEIGHT,
        model_width=MODEL_IMG_WIDTH, model_height=MODEL_IMG_HEIGHT
    )
    model = PolicyModel(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    robot = RobotController(ip=ROBOT_IP, port=ROBOT_PORT)
    robot.start_estop_listener()

    try:
        # --- 2. 相机预热与观测历史初始化 ---
        print("\n相机正在预热，请等待图像稳定...")
        for _ in range(30): 
            camera.get_latest_frame(); time.sleep(0.03)

        # 【关键修改】使用双端队列(deque)来高效地存储最近的N_OBS_STEPS帧
        obs_history = deque(maxlen=N_OBS_STEPS)

        print("正在初始化观测历史...")
        while len(obs_history) < N_OBS_STEPS:
            obs_bgr = camera.get_latest_frame()
            if obs_bgr is not None:
                obs_tensor = camera.preprocess_for_model(obs_bgr, model.device)
                obs_history.append(obs_tensor)
                print(f"已捕获 {len(obs_history)}/{N_OBS_STEPS} 帧初始图像...")
                time.sleep(0.1) # 短暂延时以获取稍微不同的帧
        
        print("="*50)
        input("初始化完成，按 Enter 键开始实时控制循环...")
        
        # --- 3. 主控制循环 ---
        print("\n开始实时控制循环。按 'S' 急停，按 Ctrl+C 退出程序。")
        while not robot.stop_flag.is_set():
            # a. 获取当前最新观测图像
            obs_bgr = camera.get_latest_frame()
            if obs_bgr is None:
                print("警告：无法获取当前帧，跳过此循环。"); time.sleep(0.1); continue
            
            cv2.imshow("Live View", obs_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # b. 图像预处理并更新观测历史队列 (自动挤出最旧的帧)
            obs_tensor = camera.preprocess_for_model(obs_bgr, model.device)
            obs_history.append(obs_tensor)
            
            # c. 模型推理，得到动作
            predicted_action_6d = model.predict_action(obs_history)
            
            # d. 获取机械臂当前位姿
            current_pose_mat = robot.get_current_pose_matrix()
            if current_pose_mat is None:
                print("无法获取当前机械臂位姿，跳过此循环。"); time.sleep(0.1); continue

            # e. 执行动作
            robot.execute_action(current_pose_mat, predicted_action_6d)

            # 控制循环频率 (可根据需要调整)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n用户通过 Ctrl+C 请求停止程序。")
    except Exception as e:
        print(f"\n程序发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 4. 清理 ---
        camera.stop()
        robot.disconnect()
        cv2.destroyAllWindows()
        print("程序已安全退出。")

if __name__ == '__main__':
    main()