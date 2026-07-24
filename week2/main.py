import pandas as pd
import numpy as np
import matplotlib
import sklearn
import sys
import cv2
import time
from collections import deque

print("python버전 :" + sys.version)                # python
print("pandas버전 :" + pd.__version__)             # pandas
print("numpy버전 :" + np.__version__)              # numpy
print("matplotlib버전 :" + matplotlib.__version__) # matplotlib
print("sklearn버전 :" + sklearn.__version__)       # sklearn
print("cv2버전 :" + cv2.__version__)               # cv2


cap = None
frame_count = 0
elapsed_time = 0.0
display_fps = 0.0
start_time = time.time()
end_time = 0.0

def to_grayscale(frame):
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray

def apply_blur(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    return blur

def detect_edges(blur):
    edges = cv2.Canny(blur, 50, 150)  # 임계값은 상황에 맞게 조정

    return edges

def preprocess(frame):
    gray = to_grayscale(frame)
    blur = apply_blur(gray)
    edges = detect_edges(blur)

    return edges

def calculate_fps():
    global frame_count, elapsed_time, display_fps, start_time, end_time

    frame_count += 1

    if frame_count == 30:
        end_time = time.time()
        elapsed_time = end_time - start_time

        display_fps = frame_count / elapsed_time

        frame_count = 0
        start_time = time.time()

    return display_fps

def draw_fps(display_fps, frame, fps):
    # FPS를 프레임에 표시
    cv2.putText(frame, f"원본 FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"30Frame_avg FPS: {display_fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def combine_frames(frame, edges):
    # combined = cv2.hconcat([frame, gray])  # 가로 결합
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([frame, edges_colored])

    return combined

def create_mask(frame):

    # BGR -> HSV로 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # pixel_bgr = frame[340, 250]   # NumPy는 [행, 열] = [y, x] 순서
    # pixel_hsv = cv2.cvtColor(pixel_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
    # print(pixel_hsv)

    # cv2.namedWindow("Mask")
    # for name, val in [("H_low",0),("S_low",120),("V_low",70),("H_high",10),("S_high",255),("V_high",255)]:
    #     cv2.createTrackbar(name, "Mask", val, 255, lambda x: None)
    #     lower = np.array([cv2.getTrackbarPos(n, "Mask") for n in ("H_low","S_low","V_low")])
    #     upper = np.array([cv2.getTrackbarPos(n, "Mask") for n in ("H_high","S_high","V_high")])
    #     mask = cv2.inRange(pixel_hsv, lower, upper)

    # 2. 추출할 색상 범위 설정 (예: 빨간색)
    # OpenCV HSV 범위: H(0~179), S(0~255), V(0~255)
    # 빨간색은 Hue 축의 양 끝(0 부근, 170~180 부근)에 걸쳐 있어 2개 영역을 합쳐줍니다.
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 3. 마스크(Mask) 생성 (범위 내의 영역은 255/흰색, 이외는 0/검은색)
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2) # 두 마스크 합치기

    cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    return mask

def find_objects(frame, mask_frame):
    # 3. Contour 검출
    # cv2.findContours는 (마스크 이미지, 검출 모드, 근사화 방법)을 인자로 받음
    contours, _ = cv2.findContours(mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 검출된 각 Contour에 대해 중심좌표 계산
    for cnt in contours:
        # 노이즈나 너무 작은 객체는 제외 (면적 기준)
        area = cv2.contourArea(cnt)
        if area > 500:  # 픽셀 면적이 500 이상인 객체만 처리
            
            # 모멘트(Moments) 계산
            M = cv2.moments(cnt)
            
            if M["m00"] != 0:
                # 중심 좌표 (cX, cY) 계산
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Contour 그리기 (초록색 윤곽선)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

                frame = draw_detection(frame, cnt, cX, cY)
                
                # 중심에 빨간색 원 그리기
                # cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                
                # 중심 좌표 텍스트 표시
                # text = f"({cX}, {cY})"
                # cv2.putText(frame, text, (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print(f"중심: ({cX}, {cY}) | 면적: {area}")

    return frame

def draw_detection(frame, cnt, cX, cY):
    # 바운딩 박스 좌표 계산 (x, y: 좌상단 좌표, w: 가로, h: 세로)
    x, y, w, h = cv2.boundingRect(cnt)

    # 시각화 그리기
    # 바운딩 박스 (초록색 직사각형, 두께 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 중심에 빨간색 원 그리기
    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    # 중심 좌표 텍스트 표시
    text = f"Center: ({cX}, {cY})"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def detect(frame):
    mask_frame = create_mask(frame)
    frame = find_objects(frame, mask_frame)

    frame = combine_frames(frame, mask_frame)

    return frame


def main():
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"웹캠 발견: index {i}")
            break
        cap.release()
    # 웹캠 대신 동영상 파일을 입력으로 사용
    print(cap.get(0))
    if -1 == cap.get(0):
        cap = cv2.VideoCapture("./sample_video.mp4")
        if cap.isOpened():
            print("테스트영상을입력으로사용")
        else:
            print("테스트영상 입력 실패")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps
    print(f'FPS: {fps}, Total frames: {total_frames}, Duration: {duration_sec}s')

    frame_time = int(1000/fps)

    # 프레임 크기(가로, 세로) 가져오기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("너비Width: " + str(w) + " 높이height: " + str(h))
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('combined.mp4', fourcc, fps, (w*2, h))  # 가로 2배 결합 예시
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # edges = preprocess(frame)

        # frame = combine_frames(frame, edges)

        display_fps = calculate_fps()
        
        frame = draw_fps(display_fps, frame, fps)

        frame = detect(frame)

        # out.write(frame)
    
        cv2.imshow('Original', frame)
        #cv2.imshow('Mask', mask_frame)

        # time.sleep(1/30)  # 약 30fps 속도로 재생되도록 지연
        if cv2.waitKey(frame_time) == ord('q'):
            break
    
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



# cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# CAP_PROP_FPS, CAP_PROP_FRAME_COUNT