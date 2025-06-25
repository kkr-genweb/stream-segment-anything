"""
WARNING: THIS FILE IS NOT CAPABLE OF RUNNING IN REALTIME YET, RETAINED TO HAVE SCAFFOLDING FOR THE FUTURE
"""


import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# ---------- Initialize SAM ----------
print("Loading SAM model...")
# checkpoint_path = "sam_vit_h_4b8939.pth" ## This is the biggest and slowest model
checkpoint_path = "sam_vit_b_01ec64.pth" ## This is the base model that is faster
device = "cuda" if torch.cuda.is_available() else "cpu"

#sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(device)

mask_generator = SamAutomaticMaskGenerator(sam)
print("SAM model loaded.")

# ---------- Frame Extraction + Segmentation ----------
def extract_frames(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(rgb_frame)


            # masks is a list of dicts — extract the mask
            mask = masks[0]["segmentation"].astype("uint8") * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Original Frame", frame)
            cv2.imshow("SAM Mask", mask_bgr)

            # Optional: save masks here

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# ---------- Main Execution ----------
if __name__ == "__main__":
    video_directory = Path("bdd100k/videos/train/")

    if not video_directory.is_dir():
        print(f"Error: Directory not found at '{video_directory}'")
    else:
        mov_files = list(video_directory.glob("*.mov"))

        if not mov_files:
            print("No .mov files found in the specified directory.")
        else:
            print(f"Found {len(mov_files)} videos to process.")

            for video_file in mov_files:
                print("-" * 60)
                print(f"Processing video: {video_file.name}")

                try:
                    extract_frames(str(video_file), frame_skip=10)
                except Exception as e:
                    print(f"Error processing {video_file.name}: {e}")
                    continue

    cv2.destroyAllWindows()
    print("All videos processed.")