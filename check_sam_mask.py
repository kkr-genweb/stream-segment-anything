import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- Mask overlay function (reused each frame) ---
def show_anns(ax, anns):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# --- Load SAM ---
# checkpoint_path = "sam_vit_h_4b8939.pth" ## This is the biggest and slowest model
checkpoint_path = "sam_vit_b_01ec64.pth" ## This is the base model that is faster
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
mask_generator = SamAutomaticMaskGenerator(sam)

# --- Load all frames ---
frame_dir = Path("frames/")
frame_files = sorted(frame_dir.glob("*.jpg"))

# --- Set up figure once ---
fig, ax = plt.subplots(figsize=(10, 10))
im_handle = None

# Turn on interactive mode
plt.ion()

for frame_path in frame_files:
    print(f"Frame: {frame_path.name}")

    image = cv2.imread(str(frame_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    ax.clear()
    ax.imshow(image)
    show_anns(ax, masks)
    ax.set_title(frame_path.name)
    ax.axis("off")
    plt.draw()
    plt.pause(0.1)  # Adjust pause to control "frame rate"

# Turn off interactive mode when done
plt.ioff()
plt.show()