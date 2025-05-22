import os
import cv2
import time
import math
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Cấu hình ---
helmet_model_path = "E:/dataset/Train-data/helment-FINAL/best.pt"
plate_model_path = "E:/dataset/license-plate-recognition/Yolov8-Detect-Vietnamese-license-plates-and-characters-main/YOLO-Weights/license_plate_detector.pt"
video_path = 'E:/imgs/6616427026658.mp4'
output_dir = 'violations'
temp_frames_dir = 'temp_frames'
location = 'HaNoi'
frame_interval = 0.5  # giây

# Tạo thư mục
Path(output_dir).mkdir(exist_ok=True)
Path(temp_frames_dir).mkdir(exist_ok=True)

# Load model
helmet_model = YOLO(helmet_model_path)
plate_model = YOLO(plate_model_path)
helmet_class_names = helmet_model.names

# Khởi tạo DeepSORT tracker
tracker = DeepSort(max_age=30)

# Theo dõi ID đã xử lý
processed_ids = set()

# --- Hàm tạo ID vi phạm ---
def generate_violation_id():
    now = datetime.now()
    date_prefix = now.strftime("%Y-%m")
    id_file = f"{output_dir}/id_{date_prefix}.txt"
    if os.path.exists(id_file):
        with open(id_file, 'r+') as f:
            current_id = int(f.read().strip()) + 1
            f.seek(0)
            f.write(str(current_id))
    else:
        current_id = 1
        with open(id_file, 'w') as f:
            f.write(str(current_id))
    return current_id

# --- Bước 1: Trích xuất frame ---
def extract_frames(video_path, output_folder, interval=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * interval)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"✅ Đã trích xuất {saved} frame.")

# --- Bước 2: Xử lý từng frame ---
def process_frames():
    for img_file in sorted(os.listdir(temp_frames_dir)):
        img_path = os.path.join(temp_frames_dir, img_file)
        frame = cv2.imread(img_path)

        results = helmet_model(frame)[0]
        detections = []

        # Tạo danh sách bbox để tracker xử lý
        for box in results.boxes:
            cls = int(box.cls)
            label = helmet_class_names[cls]
            conf = float(box.conf)

            if label in ['No-Helmet', 'None-helmet']:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, label))

        # Cập nhật tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if track_id in processed_ids:
                continue

            processed_ids.add(track_id)

            # Lưu ảnh Rider
            rider_crop = frame[y1:y2, x1:x2]
            if rider_crop.size == 0:
                continue
            rider_crop_resized = cv2.resize(rider_crop, (640, 640))

            # Tạo thư mục vi phạm
            violator_id = generate_violation_id()
            folder_name = f"{output_dir}/{violator_id}_{location}"
            os.makedirs(folder_name, exist_ok=True)

            rider_path = os.path.join(folder_name, "rider.jpg")
            cv2.imwrite(rider_path, rider_crop_resized)
            print(f"🚨 Vi phạm #{track_id} tại: {rider_path}")

            # --- Detect biển số ---
            plate_results = plate_model(rider_crop_resized)[0]
            for p_box in plate_results.boxes:
                p_class = int(p_box.cls)
                p_label = plate_model.names[p_class]
                if p_label.lower() in ['license-plate', 'licenseplate']:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                    plate_crop = rider_crop_resized[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue
                    plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    plate_path = os.path.join(folder_name, "plate.jpg")
                    cv2.imwrite(plate_path, plate_crop)
                    print(f"✅ Biển số lưu tại: {plate_path}")
                    break

# --- Chạy toàn bộ ---
extract_frames(video_path, temp_frames_dir, interval=frame_interval)
process_frames()
print("🏁 Hoàn tất.")
