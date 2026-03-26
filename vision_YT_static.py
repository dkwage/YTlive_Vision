import cv2
import yt_dlp
from ultralytics import YOLO
import supervision as sv
import csv
from datetime import datetime, timedelta
from tqdm import tqdm
import os

# ==========================================
# 1. 설정 영역
# ==========================================
YOUTUBE_URL = "https://www.youtube.com/watch?v=YoSqEM6StuY"
VIDEO_REAL_START = datetime(2026, 3, 23, 8, 40)

START_POINT = sv.Point(758, 565) 
END_POINT = sv.Point(1087, 565)  

CSV_FILENAME = "regression_data_test.csv" 
INTERVAL_MINS = 0.1

# [추가] 이미지 히트맵 데이터 기반 수업 밀집도 맵 (월~금 순서)
# 형식: "시:분": [월, 화, 수, 목, 금]
DENSITY_MAP = {
    "09:00": [108, 138, 122, 137, 94],
    "10:00": [0, 1, 0, 1, 0],
    "10:30": [144, 128, 148, 124, 41],
    "11:00": [17, 23, 13, 14, 6],
    "11:30": [0, 0, 1, 1, 1],
    "12:00": [74, 86, 79, 85, 25],
    "13:00": [49, 34, 31, 46, 30],
    "13:30": [153, 153, 167, 149, 57],
    "15:00": [151, 152, 155, 140, 53],
    "16:30": [68, 63, 70, 54, 8],
    "18:00": [24, 32, 20, 17, 6],
    "19:00": [4, 8, 11, 6, 4],
    "20:00": [10, 4, 4, 4, 2]
}

DAY_INDEX = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}

def get_class_density(day_name, time_obj):
    """
    보간치(Interpolation) 적용 버전:
    수업 시작 30분 전부터 점진적으로 증가하여 시작 정각에 최대치, 
    이후 10분 동안 서서히 감소하는 '삼각형 가중치' 모델입니다.
    """
    day_idx = DAY_INDEX.get(day_name, 0)
    current_min = time_obj.hour * 60 + time_obj.minute
    
    max_density = 0
    
    for slot_time_str, densities in DENSITY_MAP.items():
        h, m = map(int, slot_time_str.split(':'))
        slot_min = h * 60 + m
        
        peak_val = densities[day_idx]
        
        # 1. 유입 구간 (수업 시작 30분 전 ~ 정각)
        if (slot_min - 30) <= current_min <= slot_min:
            # 0에서 1로 변하는 비율(ratio) 계산
            ratio = (current_min - (slot_min - 30)) / 30
            current_val = peak_val * ratio
            max_density = max(max_density, current_val)
            
        # 2. 유출 구간 (수업 시작 정각 ~ 10분 후)
        elif slot_min < current_min <= (slot_min + 10):
            # 1에서 0으로 변하는 비율(ratio) 계산
            ratio = ((slot_min + 10) - current_min) / 10
            current_val = peak_val * ratio
            max_density = max(max_density, current_val)
            
    return round(max_density, 2)

# ==========================================

def get_vod_url(url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'compat_opts': ['remote-components'],
        'esm_location': 'github',
        'prefer_ffmpeg': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info['url']
        except Exception:
            cmd = f'yt-dlp --remote-components ejs:github -g "{url}"'
            return os.popen(cmd).read().strip()

def main():
    # [수정] CSV 헤더에 Class_Density 추가
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Day", "Time_Slot", "Class_Density", "In_Count_Delta", "Out_Count_Delta", "Unique_Total"])

    print("유튜브 연결 중...")
    stream_url = get_vod_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_interval = fps * 60 * INTERVAL_MINS 

    model = YOLO("yolo26l.pt")
    tracker = sv.ByteTrack()
    line_zone = sv.LineZone(start=START_POINT, end=END_POINT)

    prev_in, prev_out = 0, 0
    unique_ids = set()
    frame_count = 0

    with tqdm(total=total_frames if total_frames > 0 else None, desc="데이터 수집 중") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            pbar.update(1)

            results = model(frame, classes=[0], conf=0.25, device='mps', verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            line_zone.trigger(detections)

            if detections.tracker_id is not None:
                for tid in detections.tracker_id:
                    unique_ids.add(tid)

            if frame_count % frames_per_interval == 0:
                elapsed = timedelta(seconds=(frame_count / fps))
                current_time = VIDEO_REAL_START + elapsed
                day_name = current_time.strftime("%A")
                
                # [추가] 수업 밀집도 매핑
                density = get_class_density(day_name, current_time)
                
                in_delta = line_zone.in_count - prev_in
                out_delta = line_zone.out_count - prev_out
                
                with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_time.strftime("%Y-%m-%d"),
                        day_name,
                        current_time.strftime("%H:%M"),
                        density, # 자동으로 채워짐
                        in_delta,
                        out_delta,
                        len(unique_ids)
                    ])
                
                prev_in, prev_out = line_zone.in_count, line_zone.out_count

    cap.release()
    print(f"최종 저장 완료: {CSV_FILENAME}")

if __name__ == "__main__":
    main()