import argparse

import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path
import os
import math
from PIL import ImageFont, ImageDraw, Image

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
    return frame.copy()[y1:y2, x1:x2]

def drawSubFrame(frame, sub_frame, xyxy):
    x1, y1, x2, y2 = xyxy
    frame[y1:y2, x1:x2] = sub_frame
    return frame

def cal_cycler(isCycler):
    cnt = 0
    for item in isCycler:
        if item[1]:
            cnt += 1
    return cnt

def addInfoText(frame, cycler_cnt, person_cnt, bicycle_cnt, motorcycle_cnt):
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = ImageFont.truetype("SimSun.ttf", 30, encoding="utf-8")
    font_scale = 1
    font_color = (99, 28, 241)  # 紫色
    line_type = 2
    
    # text = f"people number : {cycler_cnt}\n" + f"cycler number : {cycler_cnt}\n"
    # text_size, _ = cv2.getTextSize(text, font, font_scale, line_type)
    # text_x = frame.shape[1] - text_size[0] - 10  # 10 是文字与右边界的距离
    # text_y = text_size[1] + 10  # 10 是文字与顶部边界的距离 
    # cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, line_type)
    
    total_vehicle_num = bicycle_cnt+motorcycle_cnt
    one_motorcycle_num = bicycle_cnt+2*motorcycle_cnt-cycler_cnt
    two_motorcycle_num = cycler_cnt-bicycle_cnt-motorcycle_cnt

    if total_vehicle_num == 0:
        one_motorcycle_num = 0
        two_motorcycle_num = 0
    
    if one_motorcycle_num < 0:
        one_motorcycle_num = 0
    
    if two_motorcycle_num < 0:
        two_motorcycle_num = 0

    # row1 = f"总人数: {person_cnt}"
    # row2 = f"走路: {person_cnt-cycler_cnt}"
    # row3 = f"骑行: {cycler_cnt}"

    # row4 = f"总车数: {total_vehicle_num}"
    # row5 = f"自行车: {bicycle_cnt}"
    # row6 = f"单人摩托: {one_motorcycle_num}"
    # row7 = f"双人摩托: {two_motorcycle_num}"

    # row_list = [row1, row2, row3, row4, row5, row6, row7]

    row1 = f"总人数: {person_cnt}"
    row2 = f"走路: {person_cnt-cycler_cnt}"
    row3 = f"骑行: {cycler_cnt}"

    row4 = f"总车数: {total_vehicle_num}"
    row5 = f"自行车: {bicycle_cnt}"
    row6 = f"单人摩托: {one_motorcycle_num}"
    row7 = f"双人摩托: {two_motorcycle_num}"
    
    row_list = [row1 + " " + row2 + " " + row3, row4 + " " + row5 + " " + row6 + " " + row7]

    text_size_list = []
    for row in row_list:
        # print(row)
        # text_size, _ = cv2.getTextSize(row, font, font_scale, line_type)
        text_size = font.getsize(row)
        text_size_list.append(text_size)
        
    for i in range(len(row_list)):
        row = row_list[i]
        tx = text_size_list[i][0]
        ty = text_size_list[i][1]

        x = 0.99 * frame.shape[1] - tx
        y = 0.01 * frame.shape[0] - ty * 0.2 + ty * 1.1 * i

        frame_pil = Image.fromarray(frame)  #转为PIL的图片格式
        draw = ImageDraw.Draw(frame_pil)
        ImageDraw.Draw(frame_pil).text((int(x), int(y)), row, font_color, font)
        frame = np.array(frame_pil)
    return frame

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
            txt_path = f"{base_folder}/mylabels/{p.stem}_{idx+1}.txt"
            objs = []

            person_cnt = 0
            bicycle_cnt = 0
            motorcycle_cnt = 0
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line[:-1]    # 去除最后的\n
                        cls, a, b, c, d, conf = line.split(" ")
                        
                        cls = int(cls)  # integer class
                        xyxy = [int(a), int(b), int(c), int(d)]
                        conf = float(conf)

                        if cls == 2:
                            objs.append(xyxy)
                            person_cnt += 1
                        elif cls == 0:
                            bicycle_cnt += 1
                        elif cls == 1:
                            motorcycle_cnt += 1
                    # print(f'[debug] {txt_path} 人{person_cnt} 自{bicycle_cnt} 摩{motorcycle_cnt}')

            # temp_c = 0

            # 统计每一帧内的骑车人数
            cycler_cnt = 0
            
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
                        cycler_cnt += 1
                        
                    info_list.append([idx+1, angle])
                
                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_sub_frame_{temp_c}_after.jpg', sub_frame)
                
                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_frame_{temp_c}_before.jpg', frame)

                frame = drawSubFrame(frame, sub_frame, obj)

                # if idx+1 == 28:
                #     cv2.imwrite(f'test_28_frame_{temp_c}_after.jpg', frame)

            frame = addInfoText(frame, cycler_cnt, person_cnt, bicycle_cnt, motorcycle_cnt)
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
