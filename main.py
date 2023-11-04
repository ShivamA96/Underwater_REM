import cv2

def extract_frames(video_path, interval, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps * 1000
    frame_time = 0

    while frame_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
        ret, frame = cap.read()
        if ret:
            minutes = int(frame_time / 1000 / 60)
            seconds = int(frame_time / 1000 % 60)
            output_path = f"{output_dir}/{video_path[-12::]}_frame_{minutes:02d}_{seconds:02d}.jpg"

            cv2.imwrite(output_path, frame)
        frame_time += interval * 1000

    cap.release()

video_paths = [r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\GOPR5096.MP4", r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\GOPR5098.MP4", r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\GOPR5099.MP4", r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\GOPR5100.MP4"]
interval = [3,5,7]  # in seconds
output_dirs = [r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\images\3Seconds",r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\images\5Seconds",r"C:\Users\1100a\PycharmProjects\pixelDistanceAndRealDistance\images\7Seconds"]

for video in video_paths:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps * 1000

    print(f"Video: {video} \nFPS: {fps} \nTotal Frames: {total_frames} \nDuration: {duration} ms\n")


for video_path in video_paths:
    for i in range(len(interval)):
        extract_frames(video_path, interval[i], output_dirs[i])