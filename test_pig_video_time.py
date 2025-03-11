# -*- coding: utf-8 -*-

from ultralytics import YOLO

import torch
from ultralytics import YOLO

import torch
import time
import cv2
import numpy as np
import os

# 读取配置文件中的检测范围
def read_config(file_path):
    with open(file_path, 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除空白字符
        left, top, right, bottom = map(int, line.split())  # 将字符串分割并转换为整数
    return left, top, right, bottom
    
if __name__ == '__main__':
 
    from ultralytics import YOLO
    # Load a model
    #model = YOLO(r"F:\GitHub\PIG-AI\weights\v1.0\best.pt")  # load an official model
    model = YOLO(r"F:\GitHub\PIG-AI\weights\best-detect.pt")
    # 指定要读取的文件夹路径
    folder_path = 'D:\\PigNet-main\\ultralytics-main\\datasets\\pig\images\\train'
    out_folder_path = 'D:\\PigNet-main\\test_pic\\pig_stand_out'
    
    video_path = 'rtsp://admin:HYJK2022@192.168.111.64:554/h264/ch1/main/av_stream'
    start_time = time.time()  # 记录开始时间
    cap = cv2.VideoCapture(video_path)
    end_time = time.time()  # 记录结束时间
    # 计算消耗的时间
    elapsed_time = end_time - start_time
    print(f"open camera: {elapsed_time:.4f} seconds")
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    # 定义 VideoWriter 对象以保存输出视频
    # output_path = r'D:\PigNet-main\test_pic\fire\output_video.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码方式
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 读取检测范围
    config_path = 'config.txt'
    left, top, right, bottom = read_config(config_path)
    left = 0
    top = 0
    right = 1920
    bottom = 1080

    avg_times = 0
    total_time = 0

    model_avg_times = 0
    model_total_time = 0
    ret, frame = cap.read()
    frame_rect = frame[top:bottom, left:right]
    results = model(frame_rect, save=False, imgsz=640, conf=0.01)  # predict on an image
    while cap.isOpened():
        start_time = time.time()  # 记录开始时间
        ret, frame = cap.read()
        end_time = time.time()  # 记录结束时间
        # 计算消耗的时间
        elapsed_time = end_time - start_time
        print(f"read camera: {elapsed_time:.4f} seconds")
        avg_times=avg_times+1
        total_time = elapsed_time + total_time
        print(f"average read camera time: {total_time/avg_times:.4f} seconds")

        if not ret:
            break
        frame_rect = frame[top:bottom, left:right]
        # Predict with the model
        start_time = time.time()  # 记录开始时间
        results = model(frame_rect, save=False, imgsz=640, conf=0.01)  # predict on an image
        end_time = time.time()  # 记录结束时间
        '''
        # 获取时间消耗
        preprocess_time = results.pandas().iloc[0]['preprocess']  # 预处理时间
        inference_time = results.pandas().iloc[0]['inference']    # 推理时间
        postprocess_time = results.pandas().iloc[0]['postprocess']  # 后处理时间

        print(f"Preprocess time: {preprocess_time:.2f} ms")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Postprocess time: {postprocess_time:.2f} ms")
        '''
        # 计算消耗的时间
        elapsed_time = end_time - start_time
        print(f"processing time: {elapsed_time:.4f} seconds")
        model_avg_times = model_avg_times + 1
        model_total_time += elapsed_time
        print(f"average processing time: {model_total_time/model_avg_times:.4f} seconds")

        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
        for r in results:
            for box in r.boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # (x1, y1, x2, y2)
                conf = box.conf[0]  # Confidence score
                if conf < 0.1:
                    continue
                cls = box.cls[0]  # Class index

                # Draw the bounding box on the image
                
                cv2.rectangle(frame, (int(x1+left), int(y1+top)), (int(x2+left), int(y2+top)), (255, 0, 0), 2)  # Blue box
                cv2.putText(frame, f'Class: {int(cls)}, Conf: {conf:.2f}', (int(x1+left), int(y1+top) - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Text above the box  
        #cv2.imshow("YOLOPose", frame)
        cv2.imwrite("out.jpg",frame)
        #out.write(frame)
    # 释放视频捕获对象和 VideoWriter，并关闭所有窗口
    cap.release()
    #out.release()
    cv2.destroyAllWindows()