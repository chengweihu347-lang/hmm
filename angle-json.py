import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path


def calculate_angle(a, b, c):
    """计算三个点形成的角度"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def process_video(video_path, output_folder):
    """处理单个视频并输出JSON"""
    print(f"\n处理视频: {video_path}")
    
    # 初始化MediaPipe
    mp_pose = mp.solutions.pose
    
    # 用于JSON输出的数据结构
    video_name = Path(video_path).stem
    json_data = {
        "video_info": {
            "filename": Path(video_path).name,
            "fps": 0,
            "width": 0,
            "height": 0,
            "total_frames": 0
        },
        "frames": []
    }
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return None
    
    # 获取视频元数据
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 更新JSON元数据
    json_data["video_info"]["fps"] = fps
    json_data["video_info"]["width"] = width
    json_data["video_info"]["height"] = height
    json_data["video_info"]["total_frames"] = total_frames
    
    frame_count = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # BGR转RGB用于MediaPipe处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # MediaPipe姿态检测
            results = pose.process(image)
            
            # 初始化帧数据
            frame_data = {
                "frame_number": frame_count,
                "timestamp": round(frame_count / fps, 3),
                "joints": {
                    "shoulder_angle": None,
                    "elbow_angle": None,
                    "hip_angle": None,
                    "knee_angle": None,
                    "ankle_angle": None
                }
            }
            
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 左侧关节点
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    l_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                    l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                    
                    # 右侧关节点
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                    r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                    
                    # 计算左侧关节角度
                    frame_data["joints"]["left"]["shoulder_angle"] = round(
                        calculate_angle(l_ear, l_shoulder, l_elbow), 2)
                    frame_data["joints"]["left"]["elbow_angle"] = round(
                        calculate_angle(l_shoulder, l_elbow, l_wrist), 2)
                    frame_data["joints"]["left"]["hip_angle"] = round(
                        calculate_angle(l_shoulder, l_hip, l_knee), 2)
                    frame_data["joints"]["left"]["knee_angle"] = round(
                        calculate_angle(l_hip, l_knee, l_ankle), 2)
                    frame_data["joints"]["left"]["ankle_angle"] = round(
                        calculate_angle(l_knee, l_ankle, l_heel), 2)
                    
                    # 计算右侧关节角度
                    frame_data["joints"]["right"]["shoulder_angle"] = round(
                        calculate_angle(r_ear, r_shoulder, r_elbow), 2)
                    frame_data["joints"]["right"]["elbow_angle"] = round(
                        calculate_angle(r_shoulder, r_elbow, r_wrist), 2)
                    frame_data["joints"]["right"]["hip_angle"] = round(
                        calculate_angle(r_shoulder, r_hip, r_knee), 2)
                    frame_data["joints"]["right"]["knee_angle"] = round(
                        calculate_angle(r_hip, r_knee, r_ankle), 2)
                    frame_data["joints"]["right"]["ankle_angle"] = round(
                        calculate_angle(r_knee, r_ankle, r_heel), 2)
                    
            except Exception as e:
                print(f"帧 {frame_count} 处理错误: {e}")
            
            json_data["frames"].append(frame_data)
            
            # 显示进度
            if frame_count % 30 == 0:
                print(f"已处理: {frame_count}/{total_frames} 帧", end='\r')
    
    cap.release()
    
    # 保存JSON文件
    json_output_file = os.path.join(output_folder, f"{video_name}_angles.json")
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n完成! 共处理 {frame_count} 帧")
    print(f"JSON文件已保存: {json_output_file}")
    
    return json_output_file


def main():
    # 设置文件夹路径
    folder_path = "demoag"
    
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 查找所有MP4文件
    video_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith('.mp4')]
    
    if not video_files:
        print(f"错误: 在 '{folder_path}' 文件夹中没有找到MP4文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video}")
    
    # 处理每个视频
    results = []
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        json_file = process_video(video_path, folder_path)
        if json_file:
            results.append(json_file)
    
    print("\n" + "="*50)
    print(f"所有处理完成! 共生成 {len(results)} 个JSON文件:")
    for json_file in results:
        print(f"  - {os.path.basename(json_file)}")
    print("="*50)


if __name__ == "__main__":
    main()