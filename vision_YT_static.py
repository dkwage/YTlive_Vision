import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO
import supervision as sv
import csv
from datetime import datetime, timedelta
import os
from tqdm import tqdm

# ==========================================
# 1. 설정 및 보조 함수
# ==========================================
YOUTUBE_URL = "https://www.youtube.com/watch?v=T0YY09IGYdQ"
VIDEO_REAL_START = datetime(2026, 4, 2, 14, 46) # 분석 시작 시점의 실제 시각

# 라인 좌표 (환경에 맞게 조정)
START_POINT = sv.Point(621, 382) 
END_POINT = sv.Point(1710, 382)  

CSV_FILENAME = "regression_data_1500thu.csv" 
INTERVAL_MINS = 1 # 데이터 저장 주기 (분)

# [ID 병합 및 필터링 파라미터]
MAX_MERGE_DISTANCE = 200    # 이 픽셀 이내면 동일인물로 간주
MAX_MERGE_FRAMES = 200      # 최대 몇 프레임까지 실종된 ID를 기다릴지
PERSISTENCE_THRESHOLD = 15  # 15프레임 이상 등장해야 유효한 객체로 인정

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

# ==========================================
# 2. 메인 분석 로직
# ==========================================
def main():
    # CSV 초기화 (Class_Density 컬럼 제거)
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Day", "Time_Slot", "In_Delta", "Out_Delta", "Visit_Count", "Floating_Population", "Unique_Total"])

    print(f"유튜브 연결 시도 중: {YOUTUBE_URL}")
    stream_url = get_vod_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("스트림을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_interval = int(fps * 60 * INTERVAL_MINS)

    model = YOLO("yolo26l.pt")
    
    # [트래커 설정]
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25, 
        lost_track_buffer=1500, 
        minimum_matching_threshold=0.6 
    )
    
    line_zone = sv.LineZone(start=START_POINT, end=END_POINT)

    # # --- [시각화 도구 초기화] ---
    # box_annotator = sv.BoxAnnotator()
    # label_annotator = sv.LabelAnnotator()
    # line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.5)
    # dot_annotator = sv.DotAnnotator(color=sv.Color.YELLOW, position=sv.Position.TOP_CENTER, radius=4)

    # --- [데이터 집계 변수] ---
    prev_in, prev_out = 0, 0
    interval_ids = set()     
    total_unique_ids = set()   
    
    id_frame_counts = {}     
    disappeared_ids = {}     
    id_map = {}              
    
    frame_count = 0

    with tqdm(total=total_frames if total_frames > 0 else None, desc="분석 중") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            pbar.update(1)

            # YOLO 추론 및 트래킹
            results = model(frame, classes=[0], conf=0.35, device='mps', verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            
            if detections.tracker_id is not None:
                for i, tid in enumerate(detections.tracker_id):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    real_id = id_map.get(tid, tid)
                    
                    # ID 병합 로직
                    if real_id not in total_unique_ids and tid not in id_frame_counts:
                        best_match_id = None
                        min_dist = MAX_MERGE_DISTANCE
                        
                        for d_id, (last_pos, last_f) in list(disappeared_ids.items()):
                            if frame_count - last_f > MAX_MERGE_FRAMES:
                                disappeared_ids.pop(d_id, None)
                                continue
                            
                            dist = np.linalg.norm(np.array([cx, cy]) - np.array(last_pos))
                            if dist < min_dist:
                                min_dist = dist
                                best_match_id = d_id
                        
                        if best_match_id is not None:
                            id_map[tid] = best_match_id
                            real_id = best_match_id
                            id_frame_counts[tid] = PERSISTENCE_THRESHOLD
                    
                    # 유효 ID 필터링 및 기록
                    id_frame_counts[tid] = id_frame_counts.get(tid, 0) + 1
                    if id_frame_counts[tid] >= PERSISTENCE_THRESHOLD:
                        interval_ids.add(real_id)
                        total_unique_ids.add(real_id)
                    
                    disappeared_ids[real_id] = ((cx, cy), frame_count)

            # 라인 카운팅 (머리 지점 기준)
            if len(detections) > 0:
                head_xyxy = detections.xyxy.copy()
                head_xyxy[:, 3] = head_xyxy[:, 1] 
                head_dets = sv.Detections(xyxy=head_xyxy, tracker_id=detections.tracker_id)
                line_zone.trigger(detections=head_dets)

            # # --- [실시간 시각화 프레임 구성] ---
            # annotated_frame = frame.copy()
            # line_zone_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
            # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # labels = []
            # if detections.tracker_id is not None:
            #     for tid in detections.tracker_id:
            #         rid = id_map.get(tid, tid)
            #         # 병합되었거나 이미 등록된 ID는 별표(*) 표시
            #         prefix = "*" if rid in total_unique_ids else ""
            #         labels.append(f"{prefix}#{rid}(orig:{tid})")
            
            # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            # annotated_frame = dot_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # # 상단에 누적 유니크 카운트 표시
            # cv2.putText(annotated_frame, f"Unique Total: {len(total_unique_ids)}", (50, 80), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # # 화면 크기 조정 후 출력
            # display_frame = cv2.resize(annotated_frame, (1280, 720))
            # cv2.imshow("Crowd Analysis Real-time", display_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 주기적 CSV 저장 (1분 간격)
            if frame_count % frames_per_interval == 0:
                elapsed_time = timedelta(seconds=frame_count / fps)
                current_dt = VIDEO_REAL_START + elapsed_time
                day_name = current_dt.strftime("%A")
                
                in_delta = line_zone.in_count - prev_in
                out_delta = line_zone.out_count - prev_out
                
                with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([
                        current_dt.strftime("%Y-%m-%d"),
                        day_name,
                        current_dt.strftime("%H:%M"),
                        in_delta,
                        out_delta,
                        in_delta + out_delta,
                        len(interval_ids),
                        len(total_unique_ids)
                    ])
                
                prev_in, prev_out = line_zone.in_count, line_zone.out_count
                interval_ids = set()

    cap.release()
    cv2.destroyAllWindows()
    print(f"분석 완료: {CSV_FILENAME}")

if __name__ == "__main__":
    main()