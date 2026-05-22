# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http:# www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

import cv2
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import time
import os

#is_paused = False
timestamp_file = None
frame_index = 0
# cached frames for better visualization
cached_frames = {
    'color': None
}

def setup_camera():
    """Setup camera and stream configuration"""
    pipeline = Pipeline()
    config = Config()
    device = pipeline.get_device()

    # Try to enable all possible sensors
    video_sensors = [
        OBSensorType.COLOR_SENSOR,
    ]
    sensor_list = device.get_sensor_list()
    for sensor in range(len(sensor_list)):
        try:
            sensor_type = sensor_list[sensor].get_type()
            if sensor_type in video_sensors:
                config.enable_stream(sensor_type)
        except:
            continue

    pipeline.start(config)
    return pipeline


def create_display(color_frame, width=1280, height=720):
    if color_frame is None:
        display = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        display = cv2.resize(color_frame, (width, height))
    
    #status = "PAUSED" if is_paused else "RECORDING"
    #color = (0, 0, 255) if is_paused else (0, 255, 0)
    cv2.putText(display, "Status:RECORDING", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(display, f"Written frames: {write_count}", (10, 70),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    return display

def process_color(frame):
    """Process color image"""
    frame = frame if frame else cached_frames['color']
    cached_frames['color'] = frame
    return frame_to_bgr_image(frame) if frame else None

def main():
    # Window settings
    global frame_index
    WINDOW_NAME = "MultiStream Record Viewer"
    timestamp_path = input("Enter output filename (.txt) and press Enter to start recording: ")
    color_save_dir = "D:\\data_collect\\traj\\" + timestamp_path 
    os.makedirs(color_save_dir, exist_ok=True)
    #color_save_dir = "D:\\data_collect\\traj\\" + timestamp_path + "\\images"
    timestamp_path = color_save_dir + '\\timestamp.txt'
    color_save_dir = color_save_dir + '\\images'
    #timestamp_path = file_path.replace('.bag', '_timestamp.txt')
    timestamp_file = open(timestamp_path, 'w')
    os.makedirs(color_save_dir, exist_ok=True)
    timestamp_file.write("frame_index,frame_timestamp\n")
    
      # 自动创建文件夹，已存在则不报错
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    # Initialize camera
    pipeline = setup_camera()
    #device = pipeline.get_device()
    # initialize recording
    #recorder = RecordDevice(device, file_path)
    #imu_pipeline = setup_imu()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    while True:
        # Get all frames
        frames = pipeline.wait_for_frames(100)
        #time_now = time.time()
        if not frames:
            continue
        # Process different frame types
        frame_timestamp = time.time()
        timestamp_file.write(f"{frame_index},{frame_timestamp}\n")
        frame_index += 1
        image_filename = f"color_frame_{frame_index}.png"
        image_path = os.path.join(color_save_dir, image_filename)
        color_result = process_color(frames.get_color_frame())
        processed_frames = {'color': color_result}
        cv2.imwrite(image_path, color_result)

        # create display
        display = create_display(color_result, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.imshow(WINDOW_NAME, display)

        # check exit key
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    pipeline.stop()
    #recorder = None 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
