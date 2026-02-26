from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import json
def load_calibration(path: str | Path) -> dict:
    """从 JSON 文件加载标定参数"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转回 numpy 格式（标定时用）
    for key in ["K1", "D1", "K2", "D2", "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q"]:
        if key in data:
            data[key] = np.array(data[key])
    return data
def get_undistort_rectify_maps(calib: dict) -> tuple:
    """
    生成去畸变+校正的映射表

    Returns:
        (map1_left, map2_left, map1_right, map2_right)
    """
    img_size = tuple(calib["image_size"])
    return (
        cv2.initUndistortRectifyMap(
            calib["K1"], calib["D1"], calib["R1"], calib["P1"],
            img_size, cv2.CV_32FC1,
        ),
        cv2.initUndistortRectifyMap(
            calib["K2"], calib["D2"], calib["R2"], calib["P2"],
            img_size, cv2.CV_32FC1,
        ),
    )


filename_left = Path(r"C:\Users\Lin\Desktop\RAFT-Stereo\my_test\left")
filename_right = Path(r"C:\Users\Lin\Desktop\RAFT-Stereo\my_test\right")
calib = load_calibration(r"my_test\calibration.json")
maps_left, maps_right = get_undistort_rectify_maps(calib)
_map1_left, _map2_left = maps_left
_map1_right, _map2_right = maps_right

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按下 's' 键保存图片，按下 'q' 键退出")

count = 0
while True:
    ret, frame = cap.read()
    h,w = frame.shape[:2]
    mid = int(w//2)
    if not ret:
        print("无法接收帧")
        break

    # print(frame.shape)
    
    cv2.imshow('USB Camera', frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        left_frame = frame[:,:mid,:]
        right_frame = frame[:,mid:,:]
        left_rect = cv2.remap(left_frame, _map1_left, _map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, _map1_right, _map2_right, cv2.INTER_LINEAR)
        # left_rect = cv2.cvtColor(left_rect,cv2.COLOR_BGR2GRAY)
        # right_rect = cv2.cvtColor(right_rect,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filename_left/"raw_left.png", left_frame)
        cv2.imwrite(filename_right/"raw_right.png", right_frame)
        cv2.imwrite(filename_left/"rect_left.png", left_rect)
        cv2.imwrite(filename_right/"rect_right.png", right_rect)
        print(f"已保存")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



