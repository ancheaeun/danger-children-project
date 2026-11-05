from ultralytics import YOLO
import cv2
import time
import csv
import os
from datetime import datetime

# 모델 로드
det_model = YOLO('yolov8n.pt')  # 탐지용 (사람 검출)
cls_model = YOLO('runs/classify/train/weights/best.pt')  # 분류용 (행동 인식)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# CSV 파일 준비
csv_filename = "detect_log.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "objects_detected"])

# 타이머 초기화
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed = current_time - start_time

    # 화면 표시
    cv2.imshow("Detecting...", frame)

    # 10초마다 실행
    if elapsed >= 10:
        results = det_model.predict(frame, conf=0.4, verbose=False)
        detected_info = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_info = f"{name} ({conf:.2f}) [{x1},{y1},{x2},{y2}]"
            detected_info.append(box_info)

        if detected_info:
            timestamp = datetime.now().strftime("%H:%M:%S")
            row = [timestamp, "; ".join(detected_info)]

            # CSV에 기록
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"📦 저장됨: {row}")

        # 타이머 초기화
        start_time = current_time

    # 종료 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()