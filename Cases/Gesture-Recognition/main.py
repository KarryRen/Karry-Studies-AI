# -*- coding: utf-8 -*-
# @Time    : 2025/5/22 17:33
# @Author  : Karry Ren

""""""

import cv2
import mediapipe as mp

# 强制指定使用内置摄像头（0为默认内置摄像头索引）
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS 专用参数

# 验证摄像头是否初始化成功
if not cap.isOpened():
    raise Exception("无法访问摄像头，请检查权限和设备连接")

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 定义手指索引（thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20）
finger_tips_ids = [4, 8, 12, 16, 20]

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 翻转图像以便自拍效果
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)

    fingers_up = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 画出关键点
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # 拇指
            if landmarks[finger_tips_ids[0]].x < landmarks[finger_tips_ids[0] - 1].x:
                fingers_up += 1

            # 其他四根手指
            for tip_id in finger_tips_ids[1:]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    fingers_up += 1

    # 显示结果
    cv2.putText(image, f'Fingers: {fingers_up}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
