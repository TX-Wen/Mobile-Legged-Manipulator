from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)

print(handle.id)
pose = arm.rm_get_current_arm_state()
pose = pose[1]['pose']
print("当前位姿", pose)
pose[2]-= 0.05
#print(arm.rm_movep_canfd(pose, True, 1, 60))
#print(arm.rm_movej_p(pose, 20, 0, 0, 1))
print(arm.rm_movel(pose, 2, 0, 0, 1))
arm.rm_delete_robot_arm()