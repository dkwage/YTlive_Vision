import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import csv
from datetime import datetime, timedelta
import os
from tqdm import tqdm
from supervision.geometry.core import Position

# ==========================================
# 1. 설정 및 보조 함수
# ==========================================
LOCAL_VIDEO_PATH = "asset/3.31-tue-1200.mov" 
VIDEO_REAL_START = datetime(2026, 3, 31, 11, 40)
START_POINT = sv.Point(435, 504) 
END_POINT = sv.Point(1652, 504)  
CSV_FILENAME = "regression_data.csv" 
INTERVAL_MINS = 1

# [병합 파라미터]
MAX_MERGE_DISTANCE = 200  # 이 픽셀 이내면 동일인물로 간주
MAX_MERGE_FRAMES = 200    

DENSITY_MAP = {
    "09:00": [108, 138, 122, 137, 94], "10:30": [144, 128, 148, 124, 41],
    "12:00": [74, 86, 79, 85, 25], "13:30": [153, 153, 167, 149, 57],
    "15:00": [151, 152, 155, 140, 53]
}

def get_class_density(day_name, time_obj):
    day_idx = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}.get(day_name, 0)
    current_min = time_obj.hour * 60 + time_obj.minute
    max_d = 0
    for slot_str, densities in DENSITY_MAP.items():
        h, m = map(int, slot_str.split(':'))
        slot_min = h * 60 + m
        peak = densities[day_idx]
        if slot_min - 30 <= current_min <= slot_min:
            max_d = max(max_d, peak * (current_min - (slot_min - 30)) / 30)
        elif slot_min < current_min <= slot_min + 10:
            max_d = max(max_d, peak * ((slot_min + 10) - current_min) / 10)
    return round(max_d, 2)

def main():
    if not os.path.exists(LOCAL_VIDEO_PATH): return

    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Day", "Time_Slot", "Class_Density", "In_Delta", "Out_Delta", "Visit_Count", "Floating_Population", "Unique_Total"])

    cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_per_interval = int(fps * 60 * INTERVAL_MINS)

    model = YOLO("yolo26l.pt")
    
    # [수정] 트래커 버퍼 상향 및 임계값 조정
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25, 
        lost_track_buffer=1500,          
        minimum_matching_threshold=0.6 
    )
    
    line_zone = sv.LineZone(start=START_POINT, end=END_POINT)
    
    # 시각화 도구
    # box_annotator = sv.BoxAnnotator()
    # label_annotator = sv.LabelAnnotator()
    # line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.5)
    # dot_annotator = sv.DotAnnotator(color=sv.Color.YELLOW, position=Position.TOP_CENTER, radius=4)

    # --- [데이터 집계 및 ID 병합용 변수] ---
    prev_in, prev_out = 0, 0
    interval_ids = set()
    total_unique_ids = set()
    
    id_frame_counts = {}           # ID별 누적 프레임 수
    disappeared_ids = {}           # 사라진 ID의 마지막 정보 {id: (last_center, last_frame)}
    id_map = {}                    # 병합된 ID 매핑 {new_id: original_id}
    
    PERSISTENCE_THRESHOLD = 15      # 5프레임 이상 나타나야 집계
    frame_count = 0

    print(f"분석 시작: {LOCAL_VIDEO_PATH}")

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            pbar.update(1)

            results = model(frame, classes=[0], conf=0.35, device='mps', verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            
            current_active_ids = []

            if detections.tracker_id is not None:
                for i, tid in enumerate(detections.tracker_id):
                    # 바운딩 박스 중심점 계산
                    x1, y1, x2, y2 = detections.xyxy[i]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 1. 이미 매핑된 ID라면 원래 ID 사용
                    real_id = id_map.get(tid, tid)
                    
                    # 2. 새로운 ID가 나타난 경우 병합 후보 찾기
                    if real_id not in total_unique_ids and tid not in id_frame_counts:
                        best_match_id = None
                        min_dist = MAX_MERGE_DISTANCE
                        
                        for d_id, (last_pos, last_f) in list(disappeared_ids.items()):
                            # 시간 차이 확인
                            if frame_count - last_f > MAX_MERGE_FRAMES:
                                del disappeared_ids[d_id]
                                continue
                            
                            # 거리 차이 확인
                            dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_match_id = d_id
                        
                        if best_match_id is not None:
                            id_map[tid] = best_match_id
                            real_id = best_match_id
                            # 병합된 경우 프레임 카운트 상속 (바로 필터링 통과)
                            id_frame_counts[tid] = PERSISTENCE_THRESHOLD 
                    
                    # 3. 데이터 업데이트
                    id_frame_counts[tid] = id_frame_counts.get(tid, 0) + 1
                    if id_frame_counts[tid] >= PERSISTENCE_THRESHOLD:
                        interval_ids.add(real_id)
                        total_unique_ids.add(real_id)
                    
                    current_active_ids.append(tid)
                    # 마지막 위치 업데이트 (병합용)
                    disappeared_ids[real_id] = ((cx, cy), frame_count)

            # 머리 기준 판정 및 시각화 (기존 로직 유지)
            if len(detections) > 0:
                mod_xyxy = detections.xyxy.copy()
                mod_xyxy[:, 3] = mod_xyxy[:, 1]
                head_dets = sv.Detections(xyxy=mod_xyxy, tracker_id=detections.tracker_id)
                line_zone.trigger(detections=head_dets)

            # 시각화 프레임 구성
            # annotated_frame = frame.copy()
            # line_zone_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
            # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # labels = []
            # if detections.tracker_id is not None:
            #     for tid in detections.tracker_id:
            #         rid = id_map.get(tid, tid)
            #         prefix = "*" if rid in total_unique_ids else ""
            #         labels.append(f"{prefix}#{rid}(orig:{tid})")
            
            # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            # annotated_frame = dot_annotator.annotate(scene=annotated_frame, detections=detections)
            # cv2.putText(annotated_frame, f"Unique Total: {len(total_unique_ids)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # display_frame = cv2.resize(annotated_frame, (1280, 720))
            # cv2.imshow("Crowd Analysis Debug", display_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 데이터 저장 로직 (생략 - 기존 1분 단위 저장 로직 사용)
            if frame_count % frames_per_interval == 0:
                elapsed = VIDEO_REAL_START + timedelta(seconds=frame_count / fps)
                density = get_class_density(elapsed.strftime("%A"), elapsed)
                in_d, out_d = line_zone.in_count - prev_in, line_zone.out_count - prev_out
                
                with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([elapsed.strftime("%Y-%m-%d"), elapsed.strftime("%A"), elapsed.strftime("%H:%M"), 
                                     density, in_d, out_d, in_d + out_d, len(interval_ids), len(total_unique_ids)])
                
                prev_in, prev_out = line_zone.in_count, line_zone.out_count
                interval_ids = set() 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()