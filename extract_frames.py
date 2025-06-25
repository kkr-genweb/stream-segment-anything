import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            filename = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(filename, frame)
            frame_id += 1
        count += 1
    cap.release()

# Example usage
extract_frames("bdd100k/videos/train/00dc5030-237e7f71.mov", "frames/", frame_rate=1)
