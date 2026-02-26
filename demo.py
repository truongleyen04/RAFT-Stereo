import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import json
import threading
import time

DEVICE = 'cuda'
#---------------------------------------------------------------
class CameraStream:
    def __init__(self, src=0, width=640, height=240):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock() # 线程锁，防止读写冲突

    def start(self):
        # 启动子线程执行 update
        t = threading.Thread(target=self.update, args=())
        t.daemon = True # 随主线程退出
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return
            # 更新当前帧
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
#--------------------------------------------------------------

def load_calibration(path) -> dict:
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

    

def demo(args,cam):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # output_directory = Path(args.output_directory)
    # output_directory.mkdir(exist_ok=True)
    # last = 0
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    with torch.no_grad():
        while True:
            start_time = time.time()
            ret, frame = cam.read()
            if not ret:
                print("无法接收帧") 
            h,w = frame.shape[:2]
            mid = int(w//2)
            left_frame = frame[:,:mid,:]
            right_frame = frame[:,mid:,:]
            left_rect = cv2.remap(left_frame, _map1_left, _map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_frame, _map1_right, _map2_right, cv2.INTER_LINEAR)

            image1 = torch.from_numpy(left_rect).permute(2, 0, 1).to(DEVICE, non_blocking=True).float().unsqueeze(0)
            image2 = torch.from_numpy(right_rect).permute(2, 0, 1).to(DEVICE, non_blocking=True).float().unsqueeze(0)
            if image1==None or image2==None:
                print("load_image()无效!")
                break
            

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            # file_stem = imfile1.split('/')[-2]
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
            disp = flow_up.cpu().numpy().squeeze()

            # print(f"{(last-disp[100,100])/last*100}%")
            # last = disp[100,100]
            
            disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_color = cv2.applyColorMap(-disp_vis, cv2.COLORMAP_JET)
            c_time = time.time()
            fps = 1/(c_time-start_time)
            text = f"FPS: {fps:.1f}"
            cv2.putText(disp_color,text,(50,100),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),2)
            cv2.imshow("frame",disp_color)
            
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    calib = load_calibration(r"my_test\calibration.json")
    maps_left, maps_right = get_undistort_rectify_maps(calib)
    _map1_left, _map2_left = maps_left
    _map1_right, _map2_right = maps_right
    cam = CameraStream(src=0, width=600, height=200).start()
    demo(args,cam)
