# YTlive_Vision
Analyzing people flow using YT Live
#  People Flow Tracking System (유동 인구 추적 시스템)

이 프로젝트는 YouTube Live 스트림 또는 로컬 비디오 영상에서 실시간으로 사람을 탐지하고, 특정 구역(선)을 통과하는 유동 인구(IN/OUT) 및 총 탐지 인원수를 분석하여 기록하는 파이썬 기반의 컴퓨터 비전 시스템입니다.

특히 **Apple Silicon (M4 Mac Mini)** 환경에서 GPU(`mps`)를 최대한 활용하여 프레임 드랍 없이 실시간으로 동작하도록 최적화되어 있습니다.

##  주요 기능 (Key Features)

* **실시간 객체 탐지 및 추적:** 최신 `YOLO11` 모델과 `ByteTrack` 알고리즘을 사용하여 군중 속에서도 끈질기게 사람을 식별하고 추적합니다.
* **Apple Silicon 최적화:** `device='mps'` 설정을 통해 M4 맥미니의 뉴럴 엔진/GPU 성능을 100% 활용합니다.
* **하이 앵글(Bird's-eye view) 교정:** 카메라가 위에서 아래를 내려다보는 환경에서의 오차를 줄이기 위해, 선 통과 판별 기준점을 사람의 발밑(Bottom)이 아닌 **중심점(Center)**으로 설정했습니다 (`triggering_anchors=[sv.Position.CENTER]`).
* **중복 없는 총 인원 집계:** 파이썬 `set` 자료형을 활용해 화면을 지나간 사람들의 고유 ID를 기록하여 '총 탐지 인원(Total Detected)'을 정확히 계산합니다.
* **자동 데이터 로깅:** 10분 단위로 분석 데이터를 CSV 파일(`people_flow_log.csv`)에 자동 누적 저장합니다.
* **안전 종료 시스템:** `ESC` 키를 누르면 진행 중이던 데이터를 안전하게 저장하고 프로그램을 종료합니다.

##  기술 스택 (Tech Stack)

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **AI Model:** Ultralytics YOLOv11 (`yolo11m.pt`)
* **Tracking & Analytics:** Supervision (`sv`)
* **Stream Extraction:** `yt-dlp`

