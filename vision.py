import cv2
import yt_dlp
from ultralytics import YOLO
import supervision as sv
import time
import csv
from datetime import datetime

# 1. 유튜브 링크 설정
YOUTUBE_URL = "https://www.youtube.com/watch?v=DjdUEyjx8GM"

START_POINT = sv.Point(595, 695) 
END_POINT = sv.Point(1841, 695)  

SAVE_INTERVAL = 600              # 데이터를 저장할 주기 (단위: 초). 600초 = 10분
CSV_FILENAME = "people_flow_log.csv" 

def get_stream_url(youtube_url):
    ydl_opts = {'format': 'best', 'is_live': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def main():
    # 2. 초기 세팅 (파일 생성 및 모델 로드)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Total_In", "Total_Out, Total_Detected"])  

    # 영상 스트림 연결
    stream_url = get_stream_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    
    model = YOLO("yolo26l.pt")
    
    tracker = sv.ByteTrack()
    
    # default는 BOTTOM_CENTER(발 밑), 하이 앵글에서는 CENTER(허리/가슴 높이)
    line_zone = sv.LineZone(
    start=START_POINT,
    end=END_POINT,
    triggering_anchors=[sv.Position.CENTER] 
)
    
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1.5)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()

    unique_people_ids = set()

    start_time = time.time()
    print(f"분석 시작! {SAVE_INTERVAL/60}분 단위로 {CSV_FILENAME}에 저장됩니다.")
    
    # 3. 실시간 영상 분석
    while True:
        ret, frame = cap.read() 
        if not ret:
            print("비디오 스트림이 끊겼습니다.")
            break

        results = model(frame, classes=[0], conf=0.3, device='mps', verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        line_zone.trigger(detections)

        # 3-1. 총 탐지 인원 누적 계산

        for tracker_id in detections.tracker_id:
            unique_people_ids.add(tracker_id)
            
        total_detected = len(unique_people_ids) # 현재까지 탐지된 총사람 수

        # 3-2. 10분 타이머 체크 및 데이터 자동 저장
        current_time = time.time()
        if current_time - start_time >= SAVE_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(CSV_FILENAME, mode='a', newline='') as f:
                writer = csv.writer(f)
                # 저장 항목에 total_detected 변수를 추가
                writer.writerow([timestamp, line_zone.in_count, line_zone.out_count, total_detected])
            
            print(f"[{timestamp}] 저장 완료 - IN: {line_zone.in_count}, OUT: {line_zone.out_count}, 총 인식: {total_detected}")
            start_time = current_time 

        # 3-3. 화면 시각화 
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

        # 화면 좌측 상단에 총 탐지 인원수를 노란색 텍스트로 표시
        cv2.putText(
            annotated_frame,
            f"Total Detected: {total_detected}",
            (30, 50), # 텍스트 위치 (x, y)
            cv2.FONT_HERSHEY_SIMPLEX, # 폰트
            1.2, # 글자 크기
            (0, 255, 255), 
            3 # 글자 굵기
        )

        cv2.imshow("People Flow Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 4. 종료 시 저장
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, line_zone.in_count, line_zone.out_count, len(unique_people_ids)])
    print("종료 전 최종 데이터가 저장되었습니다.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()