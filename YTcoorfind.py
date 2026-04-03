import cv2
import yt_dlp
import os

YOUTUBE_URL = "https://www.youtube.com/watch?v=T0YY09IGYdQ"

def get_stream_url(url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        # 핵심: 원격 컴포넌트(EJS) 활성화 옵션
        'compat_opts': ['remote-components'],
        'esm_location': 'github',
        # M4 맥의 경로 이슈 방지를 위해 시스템 PATH 명시 (필요시)
        'prefer_ffmpeg': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # 유튜브 보안 챌린지 해결을 위해 내부적으로 Deno/Node를 호출함
            info = ydl.extract_info(url, download=False)
            return info['url']
        except Exception as e:
            print(f"URL 추출 실패: {e}")
            # [대안] 만약 위 옵션으로도 안 되면, 직접 os.popen으로 터미널 명령어를 실행해 주소를 따옵니다.
            print("터미널 명령어로 직접 추출을 시도합니다...")
            cmd = f'yt-dlp --remote-components ejs:github -g "{url}"'
            stream_url = os.popen(cmd).read().strip()
            return stream_url if "https" in stream_url else None

def click_event(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"클릭한 좌표: sv.Point({x}, {y})")

print("스트림 주소 가져오는 중...")
cap = cv2.VideoCapture(get_stream_url(YOUTUBE_URL))
ret, frame = cap.read() 

if ret:
    cv2.imshow("Click to find coordinates", frame)
    cv2.setMouseCallback("Click to find coordinates", click_event)
    
    print("화면에서 선의 시작점과 끝점을 클릭해보세요. (종료: 아무 키나 입력)")
    cv2.waitKey(0) 

cap.release()
cv2.destroyAllWindows()