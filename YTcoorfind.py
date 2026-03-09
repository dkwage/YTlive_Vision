import cv2
import yt_dlp

YOUTUBE_URL = "https://www.youtube.com/watch?v=DjdUEyjx8GM"

def get_stream_url(youtube_url):

    ydl_opts = {'format': 'best', 'is_live': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(youtube_url, download=False)['url']

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