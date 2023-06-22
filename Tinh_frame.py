import cv2

def count_frames(video_path):
    # Mở video
    video = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có được mở thành công hay không
    if not video.isOpened():
        print("Không thể mở video.")
        return

    # Đọc và đếm số frame trong video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đóng video
    video.release()

    return total_frames

# Đường dẫn đến video
video_path = "C:/Users/84335/PycharmProjects/Doan/Detect_HAR/Dataset/CLIMB/output1.mp4"

# Tính tổng số frame của video
frame_count = count_frames(video_path)
print("Tổng số frame của video là:", frame_count)

