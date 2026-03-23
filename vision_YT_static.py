import cv2
import yt_dlp
from ultralytics import YOLO
import supervision as sv
import csv
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 1. 설정 영역
# ==========================================
# 업로드된 일반 유튜브 영상 링크를 넣으세요.
YOUTUBE_URL = "https://www.youtube.com/watch?v=YoSqEM6StuY&feature=youtu.be"

START_POINT = sv.Point(253,186) 
END_POINT = sv.Point(367,186)  

CSV_FILENAME = "people_flow_log.csv" 
# ==========================================

def get_vod_url(youtube_url):
    """
    유튜브 VOD(녹화 영상)의 실제 데이터 스트림 주소를 가져옵니다.
    라이브가 아니므로 'is_live': True 옵션을 제거했습니다.
    """
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def main():
    # CSV 헤더 생성 (영상 속 시간 기준)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_Time(Min)", "Total_In", "Total_Out", "Total_Detected"])

    print("유튜브 영상 스트림 주소를 가져오는 중...")
    stream_url = get_vod_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    
    # 영상 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # 만약 fps를 못 가져오면 기본값 30으로 설정
    
    # 네트워크 스트림의 경우 전체 프레임 수를 정확히 못 가져올 수 있으므로 예외 처리
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    print("분석을 시작")
    
    # ==========================================
    # 3. 고속 영상 분석 루프 (진행 상태바 적용)
    # ==========================================
    # 전체 프레임 수를 알면 퍼센트(%)로, 모르면 단순 카운트로 상태바 표시
    pbar_total = total_frames if total_frames > 0 else None
    
    with tqdm(total=pbar_total, desc="영상 분석 진행률") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            pbar.update(1) # 상태바 1칸 진행

            # 화면 출력 코드를 모두 제거하고 오직 연산만 수행
            results = model(frame, classes=[0], conf=0.3, device='mps', verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            line_zone.trigger(detections)

            for tracker_id in detections.tracker_id:
                unique_people_ids.add(tracker_id)

            # 영상 속 시간 기준으로 10분(600초) 분량이 분석될 때마다 CSV에 기록
            if frame_count % frames_per_10_min == 0:
                current_video_min = int((frame_count / fps) / 60)
                
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

    print("\nCSV 파일에 저장되었습니다!")
    cap.release()

if __name__ == "__main__":
    main()