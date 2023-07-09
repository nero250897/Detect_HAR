import cv2
import mediapipe as mp
import pandas as pd
import time

# Đọc ảnh từ webcam
#url = "rtsp://admin:L2AC931D@192.168.1.99:554/cam/realmonitor?channel=1&subtype=00"
#cap = cv2.VideoCapture(url)

cap = cv2.VideoCapture('C:/Users/84335/PycharmProjects/Human_Pose/Dataset/CLIMB/output.mp4')

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "CLIMB"
no_of_frames = 2500

# Khởi tạo biến để tính toán FPS
start_time = time.time()
frame_count = 0

def make_landmark_timestep(results):
    #print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    # resize video xuat tu cam
    #frame = cv2.resize(frame, (700, 500))
    if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

            # Tính toán FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Write vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")

cap.release()
cv2.destroyAllWindows()