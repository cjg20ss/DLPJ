import argparse

import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path
import os
import math

from videosource import FileSource, WebcamSource
def cal_angle(ankle, knee, hip):
    x_ankle, y_ankle = ankle[:2]
    x_knee, y_knee = knee[:2]
    x_hip, y_hip = hip[:2]
    vec1 = [x_knee - x_ankle, y_knee - y_ankle]
    vec2 = [x_knee - x_hip, y_knee - y_hip]
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    angle_radians = math.acos(dot / (magnitude1 * magnitude2))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def getSubFrame(frame, xyxy):
    x1, y1, x2, y2 = xyxy
    # print(frame.copy()[y1:y2, x1:x2].shape)
    return frame.copy()[y1:y2, x1:x2]

def drawSubFrame(frame, sub_frame, xyxy):
    x1, y1, x2, y2 = xyxy
    frame[y1:y2, x1:x2] = sub_frame
    # print(frame[y1:y2, x1:x2].shape)
    return frame

def cal_cycler(isCycler):
    cnt = 0
    for item in isCycler:
        if item[1]:
            cnt += 1
    return cnt

def addInfoText(frame, isCycler):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2
    
    cycler_cnt = 0
    people_cnt = 0
    for item in isCycler:
        if item[1]:
            cycler_cnt += 1
        people_cnt += 1
    
    text = f"people number : {cycler_cnt}\n" + f"cycler number : {cycler_cnt}\n"
    text_size, _ = cv2.getTextSize(text, font, font_scale, line_type)
    text_x = frame.shape[1] - text_size[0] - 10  # 10 是文字与右边界的距离
    text_y = text_size[1] + 10  # 10 是文字与顶部边界的距离 
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, line_type)

    
    
    
    

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3, color=(238,241,28))
# color是BGR

def main(inp, mylabels_path='video_obj_detect'):
    mylabels_path = str(mylabels_path)
    if not os.path.exists(mylabels_path):
        print('error: you should use yolov5 to do object detection first!')
        return
    
    p = Path(str(inp))
    base_folder = f"{mylabels_path}/{p.stem}"

    if not os.path.exists(base_folder):
        print('error: you should use yolov5 to do object detection first!')
        return
    
    if inp is None:
        source = WebcamSource()
    else:
        source = FileSource(inp)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        
        fps = source._capture.get(cv2.CAP_PROP_FPS)
        w = int(source._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(source._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        p = Path(inp)

        save_dir = f"{base_folder}/video_output/pose"
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)
        os.mkdir(save_dir)
        save_path = f"{save_dir}/{p.name}"
            
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        # print(h, w)

        info_list = []

        # 遍历每一帧
        for idx, (frame, frame_rgb) in enumerate(source):
            # txt_path = f"../yolov5/myfolder/mylabels/test_{idx+1}.txt"
            txt_path = f"{base_folder}/mylabels/test_{idx+1}.txt"
            objs = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line[:-1]    # 去除最后的\n
                        cls, a, b, c, d, conf = line.split(" ")
                        
                        cls = int(cls)  # integer class
                        xyxy = [int(a), int(b), int(c), int(d)]
                        conf = float(conf)

                        if cls == 0:  # 默认的coco.yaml中，0表示person这一类别
                            objs.append(xyxy)

            # temp_c = 0

            # 统计每一帧内人的是否骑车
            isCycler = []
            
            for obj in objs:
                # temp_c += 1

                # print(f"[debug] test_{idx+1}.txt: {obj}")
                sub_frame = getSubFrame(frame, obj)
                sub_frame_rgb = getSubFrame(frame_rgb, obj)

                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_sub_frame_{temp_c}_before.jpg', sub_frame)

                # 以sub_frame_rgb作为输入，经过关键点检测之后，把检测点绘制在sub_frame之上，返回sub_frame
                results = pose.process(sub_frame_rgb)            
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        sub_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )
                    # get landmarks as numpy
                    landmarks = results.pose_landmarks.landmark
                    np_landmarks = np.array(
                        [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
                    )

                    ankle = np_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    knee = np_landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    hip = np_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    angle = cal_angle(ankle, knee, hip)
                    if angle < 140:
                        isCycler.append([obj, True])
                    else:
                        isCycler.append([obj, False])
                        
                    info_list.append([idx+1, angle])
                
                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_sub_frame_{temp_c}_after.jpg', sub_frame)
                
                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_frame_{temp_c}_before.jpg', frame)

                frame = drawSubFrame(frame, sub_frame, obj)

                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_frame_{temp_c}_after.jpg', frame)

            addInfoText(frame, isCycler)
            source.show(frame)
            vid_writer.write(frame)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose video file otherwise webcam is used."
    )
    parser.add_argument(
        "-i", metavar="path-to-file", type=str, help="Path to video file"
    )
    parser.add_argument('-a', type=str, default='video_obj_detect', help='ask cjg and csy')


    args = parser.parse_args()

    main(args.i, args.a)
