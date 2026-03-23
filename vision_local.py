import cv2
from ultralytics import YOLO
import supervision as sv
import csv
from datetime import datetime
from tqdm import tqdm # 터미널 진행 상태바 라이브러리

# ==========================================
# 1. 설정 영역
# ==========================================
VIDEO_PATH = "test.mp4" # 로컬 테스트 영상 경로
START_POINT = sv.Point(100, 300)
END_POINT = sv.Point(800, 300)

CSV_FILENAME = "fast_people_flow_log.csv"
# ==========================================

def main():
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_Time(Min)", "Total_In", "Total_Out", "Total_Detected"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 영상의 전체 프레임 수와 FPS(초당 프레임) 정보 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 영상 기준 10분(600초)에 해당하는 프레임 수 계산
    frames_per_10_min = fps * 600 

    model = YOLO("yolo26l.pt")
    tracker = sv.ByteTrack()
    
    line_zone = sv.LineZone(
        start=START_POINT,
        end=END_POINT,
        triggering_anchors=[sv.Position.CENTER]
    )

    unique_people_ids = set()
    frame_count = 0

    print(f"분석 시작 (총 {total_frames} 프레임)")
    
    # ==========================================
    # 3. 고속 영상 분석 루프 (그리기/출력 완전 제거)
    # ==========================================
    # tqdm으로 진행률(%) 바 생성
    for _ in tqdm(range(total_frames), desc="분석 진행률"):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1

        # 화면에 그리는(Annotate) 코드를 싹 다 지웠습니다. 오직 연산만 합니다.
        results = model(frame, classes=[0], conf=0.3, device='mps', verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        line_zone.trigger(detections)

        for tracker_id in detections.tracker_id:
            unique_people_ids.add(tracker_id)

        # 현실 시간이 아닌 '영상 속 시간(프레임)' 기준으로 10분어치 분량이 분석될 때마다 저장
        if frame_count % frames_per_10_min == 0:
            current_video_min = int((frame_count / fps) / 60) # 영상의 현재 분(Minute)
            
            with open(CSV_FILENAME, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{current_video_min}m", line_zone.in_count, line_zone.out_count, len(unique_people_ids)])

    # ==========================================
    # 4. 분석 완료 후 최종 결과 저장
    # ==========================================
    final_video_min = int((frame_count / fps) / 60)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"{final_video_min}m (End)", line_zone.in_count, line_zone.out_count, len(unique_people_ids)])

    print("\n✅ 고속 분석이 완료되어 데이터가 저장되었습니다!")
    cap.release()

if __name__ == "__main__":
    main()