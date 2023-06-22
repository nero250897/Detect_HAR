import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
from playsound import playsound

label = "Warmup...."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")

# Đoc anh tu webcam
#url = "rtsp://admin:L2AC931D@192.168.1.99:554/cam/realmonitor?channel=1&subtype=00"
#cap = cv2.VideoCapture(url)

cap = cv2.VideoCapture("C:/Users/84335/PycharmProjects/Doan/Detect_HAR/Video/Video_1.mp4")

# Khoi tao bien de tinh toan FPS
start_time = time.time()
frame_count = 0

# Khoi tao cac thong so cho pyaudio
audio_file_path = "C:/Users/84335/PycharmProjects/Doan/alert.wav"

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    #print(results)
    if results[0][0] > 0.5:
        label = "CLIMB"
        text = "Co xam nhap"
        # Hien canh bao len man hinh
        cv2.putText(img, "Canh bao: {}".format(text), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        playsound(audio_file_path)
    else:
        label = "UNLOCK THE DOOR"
        text = "Co xam nhap"
        # Hien canh bao len man hinh
        cv2.putText(img, "Canh bao: {}".format(text), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        playsound(audio_file_path)
    return label


i = 0
warmup_frames = 10

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    # Tinh toan FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img, f"FPS: {round(fps, 2)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
