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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('combined.mp4', fourcc, fps, (w*2, h))  # 가로 2배 결합 예시
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        edges = preprocess(frame)

        frame = combine_frames(frame, edges)

        display_fps = calculate_fps()
        
        frame = draw_fps(display_fps, frame, fps)

        out.write(frame)
    
        cv2.imshow('win', frame)
        # time.sleep(1/30)  # 약 30fps 속도로 재생되도록 지연
        if cv2.waitKey(frame_time) == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



# cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# CAP_PROP_FPS, CAP_PROP_FRAME_COUNT