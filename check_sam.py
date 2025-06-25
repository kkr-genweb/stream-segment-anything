# check_sam.py (to confirm basic model is working)
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# Load model
# checkpoint_path = "sam_vit_h_4b8939.pth" ## This is the biggest and slowest model
checkpoint_path = "sam_vit_b_01ec64.pth" ## This is the base model that is faster
#sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# Dummy image
image = np.ones((512, 512, 3), dtype=np.uint8) * 255
cv2.circle(image, (256, 256), 100, (0, 0, 255), -1)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set image
predictor.set_image(image_rgb)

# Provide a prompt: a center point
input_point = np.array([[256, 256]])
input_label = np.array([1])

# Run prediction
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# Output
print(f"Generated {len(masks)} masks")
for i, score in enumerate(scores):
    print(f"Mask {i} — Confidence score: {score:.4f}")